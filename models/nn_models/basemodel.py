import torch
from torch import nn

__all__ = ['BaseModel']


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def change_mode(self):
        raise NotImplementedError

    def measure_gradient_projection(self):
        raise NotImplementedError

    def regularization_loss(self):
        raise NotImplementedError

    def update_basis(self, conceptor_list:list):
        raise NotImplementedError

    # def update_previous_weights(self):
    #     raise NotImplementedError

    def forward(self, x, labels=None, experience_id=None):
        raise NotImplementedError

    def forward_all_layers(self, x, experience_id=None):
        raise NotImplementedError

