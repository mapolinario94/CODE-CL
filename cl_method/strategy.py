import torch
import torch.optim as optim
from avalanche.evaluation.metrics import Forgetting
from torch.utils.data import DataLoader
from utils.metrics import *
import time
import mlflow
import logging
from copy import deepcopy
from models.nn_models import BaseModel
from cl_method import code_cl

__all__ = ["MyStrategy", "average_forgetting_metric"]

class MyStrategy():

    def __init__(self, model:BaseModel, optimizer, criterion, epochs, batch_size, threshold=[0.95, 0.99, 0.99], lr=0.1,
                 aperture=4, dropout=False, data_aug=False, basis_bs=125, avg_pool=False,
                 transform_test=None, transform_train=None, dataset_name=None, model_name=None, print_freq=50,
                 aperture_gain=1.0, patience=6, lr_decay=2, lr_threshold=1e-5, lower_bound=0.2):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.experience_id = 0
        self.conceptor_list = []
        self.threshold = threshold
        self.forgetting_metric = Forgetting()
        self.feedback_list = []
        self.aperture = aperture
        self.data_aug = data_aug
        self.dropout = dropout
        self.basis_bs = basis_bs
        self.avg_pool = avg_pool
        self.transform_test = transform_test
        self.transform_train = transform_train
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.aperture_gain = aperture_gain
        self.patience = patience
        self.lr_decay = lr_decay
        self.lr_threshold = lr_threshold
        self.lower_bound = lower_bound

    def update_basis(self, experience, exp_id):
        if hasattr(experience, "dataset"):
            train_dataset = experience.dataset
        else:
            train_dataset = experience

        batch_size = self.basis_bs
        train_data_loader = DataLoader(
            train_dataset, num_workers=4, batch_size=batch_size, shuffle=True
        )

        mat_list = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(train_data_loader):
                activations = self.model.forward_all_layers(inputs.cuda(), exp_id)
                [mat_list.append(x) for x in activations]
                break
            self.conceptor_list = code_cl.update_basis(mat_list, conceptor_list=self.conceptor_list,
                                                       threshold=self.threshold, aperture=self.aperture,
                                                       model_name=self.model_name, memory_threshold=self.aperture_gain,
                                                       lower_sval_bound=self.lower_bound)
        self.model.update_basis(self.conceptor_list)

    def criterion(self, output, target):
        return self.loss_fn(output, target)

    def task_similarity(self, dataloader, optimizer, task_id):
        mat_list = []
        self.model.eval()
        features_list_new = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                activations = self.model.forward_all_layers(inputs.cuda(), task_id)
                [mat_list.append(x) for x in activations]
                break
            features_list_new = code_cl.update_basis(mat_list, conceptor_list=features_list_new,
                                                     threshold=self.threshold, aperture=self.aperture,
                                                     model_name=self.model_name, memory_threshold=self.aperture_gain,
                                                     lower_sval_bound=0.2, print_logs=False)
        self.model.measure_tasks_similarity(task_id, features_list_new)

    def train(self, experience, test_experience=None, experience_zero=None):
        if hasattr(experience, "dataset"):
            train_dataset = experience.dataset
        else:
            train_dataset = experience

        train_data_loader = DataLoader(
            train_dataset, num_workers=4, batch_size=self.batch_size, shuffle=True
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0, weight_decay=0)

        logging.info("#" * 20)
        logging.info(f"Training on task: {self.experience_id}")
        logging.info("#" * 20)
        best_model = get_model(self.model)
        best_loss = 1000
        best_acc = 0
        flag = False
        lr_updated = self.lr

        # Evaluate gradients
        if self.experience_id != 0:
            conceptor_data_loader = DataLoader(
                train_dataset, num_workers=4, batch_size=self.basis_bs, shuffle=True
            )
            self.task_similarity(conceptor_data_loader, optimizer, task_id=self.experience_id)
            conceptor_data_loader = None

        for epoch in range(self.epochs):
            if self.dropout:
                self.model.train()
            else:
                self.model.eval()
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(train_data_loader),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            end = time.time()
            for batch_idx, (inputs, labels) in enumerate(train_data_loader):
                data_time.update(time.time() - end)
                batch_size = inputs.size(0)
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs, labels, self.experience_id)

                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                if self.experience_id != 0:
                    self.model.update_gradient()
                optimizer.step()

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                losses.update(loss.item(), batch_size)
                batch_time.update(time.time() - end)
                end = time.time()
                if batch_idx % self.print_freq == (self.print_freq - 1):
                    progress.display(batch_idx, log=True)
            logging.info(f"Testing on current experience {self.experience_id} at epoch {epoch}:")
            valid_acc = self.testing(test_experience)
            # logging.info("Testing on first experience:")
            # _ = self.testing(experience_zero, test_id=0)
            with torch.no_grad():
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = get_model(self.model)
                    patience = 0
                else:
                    patience += 1
                    if patience > self.patience:
                        adjust_learning_rate(optimizer, epoch, None)
                        lr_updated /= self.lr_decay
                        patience = 0
                    elif lr_updated < self.lr_threshold:
                        break
        set_model_(self.model, best_model)
        self.experience_id += 1
        # self.model.update_previous_weights()

    def testing(self, experience, test_id=None):
        if hasattr(experience, "dataset"):
            eval_dataset = experience.dataset
        else:
            eval_dataset = experience
        eval_data_loader = DataLoader(
            eval_dataset, num_workers=4, batch_size=512
        )
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (inputs, labels) in enumerate(eval_data_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                batch_size = inputs.size(0)
                outputs = self.model(inputs, experience_id=self.experience_id if test_id is None else test_id)
                loss = self.criterion(outputs, labels)
                losses.update(loss.item(), batch_size)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
            logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg}'.format(top1=top1, top5=top5, loss=losses))
        return top1.avg.item()

    def eval(self, experience, exp_id, exp_test_id):
        if hasattr(experience, "dataset"):
            eval_dataset = experience.dataset
        else:
            eval_dataset = experience
        eval_data_loader = DataLoader(
            eval_dataset, num_workers=4, batch_size=512
        )
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (inputs, labels) in enumerate(eval_data_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                batch_size = inputs.size(0)
                outputs = self.model(inputs, experience_id=exp_test_id)
                # loss = criterion(outputs, labels)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
            logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        self.forgetting_metric.update(k=f"Experience {exp_test_id}", v=top1.avg.item(),
                                      initial=exp_test_id == (self.experience_id - 1))
        mlflow.log_metric(f"acc_exp_{exp_test_id}", top1.avg.item(), step=exp_id)
        return top1.avg.item()


def average_forgetting_metric(forgetting_dict):
    avg_forgetting = 0
    num_exp = 0
    for key in forgetting_dict.keys():
        num_exp += 1
        avg_forgetting += forgetting_dict[key]
    if num_exp == 0:
        return 0
    else:
        avg_forgetting /= num_exp
        return avg_forgetting


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        param_group['lr']=param_group['lr']/2