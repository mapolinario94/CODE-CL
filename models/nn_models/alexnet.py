import torch
from torch import nn
from .basemodel import BaseModel
from .layers import *

__all__ = ['AlexNet']


class AlexNet(BaseModel):

    def __init__(self, n_classes=10, n_experiences=3, threshold_linear=0.95, threshold_conv=0.95, num_free_dim=0):
        super(AlexNet, self).__init__()

        hidden_size = 100
        mode = "IN"
        # n_experiences = 3
        self.conv1 = CustomConv2d(3, 64, 4, bias=False, mode=mode, threshold=threshold_conv,
                                  weight_normalization=False, num_free_dim=num_free_dim)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv2 = CustomConv2d(64, 128, 3, bias=False, mode=mode, threshold=threshold_conv,
                                  weight_normalization=False, num_free_dim=num_free_dim)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv3 = CustomConv2d(128, 256, 2, bias=False, mode=mode, threshold=threshold_conv,
                                  weight_normalization=False, num_free_dim=num_free_dim)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)

        self.fc1 = CustomLinear(256 * 4, 2048, bias=False, mode=mode, threshold=threshold_linear,
                                weight_normalization=False, num_free_dim=num_free_dim)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = CustomLinear(2048, 2048, bias=False, mode=mode, threshold=threshold_linear,
                                weight_normalization=False, num_free_dim=num_free_dim)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)

        self.fc3 = torch.nn.ModuleList()
        for t in range(n_experiences):
            self.fc3.append(torch.nn.Linear(2048, n_classes, bias=False))

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.flag = False

    def measure_tasks_similarity(self, task_id, conceptor_list):
        self.conv1.measure_tasks_similarity(task_id, conceptor_list[0])
        self.conv2.measure_tasks_similarity(task_id, conceptor_list[1])
        self.conv3.measure_tasks_similarity(task_id, conceptor_list[2])
        self.fc1.measure_tasks_similarity(task_id, conceptor_list[3])
        self.fc2.measure_tasks_similarity(task_id, conceptor_list[4])


    def update_basis(self, conceptor_list: list):
        self.conv1.update_basis(basis_in=conceptor_list[0])
        self.conv2.update_basis(basis_in=conceptor_list[1])
        self.conv3.update_basis(basis_in=conceptor_list[2])
        self.fc1.update_basis(basis_in=conceptor_list[3])
        self.fc2.update_basis(basis_in=conceptor_list[4])

    # def update_previous_weights(self):
    #     self.conv1.update_previous_weights()
    #     self.conv2.update_previous_weights()
    #     self.conv3.update_previous_weights()
    #     self.fc1.update_previous_weights()
    #     self.fc2.update_previous_weights()

    def update_gradient(self):
        self.conv1.update_gradient()
        self.conv2.update_gradient()
        self.conv3.update_gradient()
        self.fc1.update_gradient()
        self.fc2.update_gradient()
        self.bn1.zero_grad()
        self.bn2.zero_grad()
        self.bn3.zero_grad()
        self.bn4.zero_grad()
        self.bn5.zero_grad()

    def forward(self, x, labels=None, experience_id=None):
        x = self.conv1(x, task_id=experience_id)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))
        x = self.conv2(x, task_id=experience_id)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))
        x = self.conv3(x, task_id=experience_id)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x, task_id=experience_id)
        x = self.drop2(self.relu(self.bn4(x)))
        x = self.fc2(x, task_id=experience_id)
        x = self.drop2(self.relu(self.bn5(x)))
        y_pred = self.fc3[experience_id](x)
        return y_pred

    def forward_all_layers(self, x0, experience_id=None):
        x = self.conv1(x0, task_id=experience_id)
        x1 = self.maxpool(self.drop1(self.relu(self.bn1(x))))
        x = self.conv2(x1, task_id=experience_id)
        x2 = self.maxpool(self.drop1(self.relu(self.bn2(x))))
        x = self.conv3(x2, task_id=experience_id)
        x3 = self.maxpool(self.drop2(self.relu(self.bn3(x))))
        x3 = x3.view(x3.size(0), -1)
        x = self.fc1(x3, task_id=experience_id)
        x4 = self.drop2(self.relu(self.bn4(x)))
        x = self.fc2(x4, task_id=experience_id)
        x5 = self.drop2(self.relu(self.bn5(x)))
        return x0, x1, x2, x3, x4, x5

