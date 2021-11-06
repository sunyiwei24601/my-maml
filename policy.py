import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import Independent, Normal
from functools import reduce
from operator import mul


def get_policy(env, hidden_sizes=(256, 512)):
    """
    extract env's input and output, create a policy object
    :param env: the gym environment to use
    :param hidden_sizes: hidden layer sizes of Policy
    :return: a Normal Policy Object
    """
    input_size = reduce(mul, env.observation_space.shape, 1)
    output_size = reduce(mul, env.action_space.shape, 1)
    return NormalPolicy(input_size, output_size, hidden_sizes=hidden_sizes)


class NormalPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinear=nn.ReLU, min_log_std=1e-6):
        super(NormalPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlinear = nonlinear



        linear_sizes = [input_size, *hidden_sizes, output_size]
        linear_layers = []
        for i in range(len(linear_sizes) - 1):
            i_size, o_size = linear_sizes[i], linear_sizes[i+1]
            linear_layers.append(nn.Linear(i_size, o_size))
            linear_layers.append(nonlinear())
        linear_layers = linear_layers[:-1]  # remove last nonlinear layer

        self.mean_policy = nn.Sequential(*linear_layers)
        self.log_std = np.ones(output_size, dtype=np.float32)
        # convert a tensor data into Parameter data, which can be updated automatically
        self.log_std = nn.Parameter(torch.Tensor(self.log_std))
        self.min_log_std = min_log_std

    def forward(self, input):
        mean = self.mean_policy(input)
        print(mean.requires_grad)
        std = torch.exp(torch.clamp(self.log_std, min=self.min_log_std))

        # Independent change the batch_shape into 1
        # distribution can be seen as batch and event, one batch contains several event
        # in this problem 3x1 -> 1x3, the result doesn't matter
        return Independent(Normal(loc=mean, scale=std), 1)

    def get_copy(self):
        """
        create a new policy object with the same parameters as current policy
        :return:
        """
        new_policy = copy.deepcopy(self)
        return new_policy




