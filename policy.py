import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import Independent, Normal
from functools import reduce
from operator import mul
from collections import OrderedDict

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
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinear=F.relu, min_log_std=1e-6):
        super(NormalPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlinear = nonlinear
        self.layer_num = len(hidden_sizes) + 1


        linear_sizes = [input_size, *hidden_sizes, output_size]
        linear_layers = []
        for i in range(len(linear_sizes) - 1):
            i_size, o_size = linear_sizes[i], linear_sizes[i+1]
            linear_layers.append(nn.Linear(i_size, o_size))

        self.mean_policy = nn.Sequential(*linear_layers)
        self.log_std = np.ones(output_size, dtype=np.float32)
        # convert a tensor data into Parameter data, which can be updated automatically
        self.log_std = nn.Parameter(torch.Tensor(self.log_std))
        self.min_log_std = min_log_std

    def forward(self, input, params=None):
        """
        rewrite the forward function. if given params, use this param to create distribution, else use the module's own parameter
        :param input:
        :param params: OrderedDict(policy.named_parameters())
        :return:
        """
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(self.layer_num):
            output = F.linear(output, weight=params['mean_policy.{}.weight'.format(i)],
                              bias=params['mean_policy.{}.bias'.format(i)])
            output = self.nonlinear(output)
        mean = output

        # generate the variance from parameter
        std = torch.exp(torch.clamp(params['log_std'], min=self.min_log_std))

        # Independent change the batch_shape into 1
        # distribution can be seen as batch and event, one batch contains several event
        # in this problem 3x1 -> 1x3, the result doesn't matter
        return Independent(Normal(loc=mean, scale=std), 1)

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        """

        :param loss:
        :param params:  OrderedDict(policy.named_parameters())
        :param step_size:
        :param first_order: if True, we will only use first order of the grad, so we will not create-graph on grad
        :return:
        """

        if params is None:
            params = OrderedDict(self.named_parameters())

        """
        we got 2 ways to get grad, backward() and autograd.grad()
        The difference is, backward will calculate the gradient and save it on x.grad
        y.backward() = torch.autograd.backward(y)
        In this way, if we have a batch of loss y, we can add the gradient together on x.grad

        On the other hand, grad = torch.autograd.grad(loss, x) will get the grad rather than update x.grad
        In this way we can calculate second-order or higher gradient with retrain_graph or create_graph 

        """
        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)

        """
        Calculate the grad manually which can be used as update direction
        Here we set create_graph=True to create a new graph after calculate the grad.
        So we can calculate the second-order gradient

        there are two parameters create_graph and retain_graph
        retain_graph=True will remain loss function's graph after calculating the grad.(normally we will clean the graph)
        create_graph=True will create a new graph between x and the first-order grad calculated so we can calculate 
            the grad of first-order grad on x
        """


        
        updated_params = OrderedDict()
        # according MAML paper, here we only update one step on params, but TODO the step size here can be modified
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params

