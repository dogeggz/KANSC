# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: MutuInfo.py
@Time: 2021/4/1 9:46
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class Mine(nn.Module):
    # Mine network with three linear layers
    # ISSUE: dimensions be 256 instead of 2, 10? and why is the ouput dimension is 1?
    #
    def __init__(self, in_dim=2, hidden_size=10):
        super(Mine, self).__init__()

        self.dense1 = linear(in_dim, hidden_size)
        self.dense2 = linear(hidden_size, hidden_size)
        self.dense3 = linear(hidden_size, 1)

    # forward pass, processes input throught three layers with relu activations between layers, transform the input into a form suitable for mutual information estimation
    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.relu(x)
        # NOTE: relu
        """
        Relu inrroduces non-linearity to the model. allows the model to learn complex patterns in the data.
        simple thresholding at zero,
        mitigate the vanishing gradient problem
        """
        x = self.dense2(x)
        x = F.relu(x)
        output = self.dense3(x)

        return output


# helper function for creating linear layers
def linear(in_dim, out_dim, bias=True):
    """
    Creates a linear layer with specified input and output dimensons
    """
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin.weight = torch.nn.Parameter(torch.normal(0.0, 0.02, size=lin.weight.shape))
    # xavier_uniform_(lin.weight)
    # bias to zeor if bias
    if bias:
        lin.bias.data.zero_()

    return lin


def mutual_information(joint, marginal, mine_net):
    # NOTE: joint distribution means: p(x,y) over the pair of variables
    # NOTE: marginal: looking at one variable's distribution regardless of the other (e.g. p(x) or p(y)
    """
    Estimates mutual information using hte mine_net
    joint: joint distribution of inptus
    marginal: marginal distribution of inputs
    mine_net: the mine network instance
    """
    t = mine_net(joint)  # output of the mine_net when applied to the joint distribution
    et = torch.exp(
        mine_net(marginal)
    )  # exponential of the output when applied to the marginal distribution
    mi_lb = torch.mean(t) - torch.log(
        torch.mean(et)
    )  # lower bound of the mutual information, calculated as the difference between the mean of t and the log of the mean of et

    return mi_lb, t, et


def learn_mine(batch, mine_net, ma_et, ma_rate=0.01):
    """
    Trains the mine network to estimate mutual information
    batch: tuple of (joint, marginal) samples
    mine_net: the mine network instance
    ma_et: movign average of et
    ma_rate: rate at which moving average formula
    """
    # NOTE: moving average: statistial technique that calcualtes the average of a subset of datapoints by updating it as new data points become available.
    # NOTE: ma_et: moving average of the exponential term `et` calculated in the mutual information function, used to maintain a running average of the `et` values, which helps in unbiasing the loss estimation
    # NOTE: ma_rate: controls the rate at which the moving average is updated, determines how much weight to give to the new value compared to the old moving average. Higher value meanst he moving average will react more quickly to changes

    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = torch.FloatTensor(joint)
    marginal = torch.FloatTensor(marginal)
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)  # moving average of et

    # unbiasing use moving average
    loss = -(
        torch.mean(t) - (1 / torch.mean(ma_et)) * torch.mean(et)
    )  # calculate teh loss as the negative of the mutual information lower bound
    # use biased estimator
    # loss = - mi_lb
    return loss, ma_et, mi_lb


def sample_batch(rec, noise):
    """
    prepares batches for training the mutual information estimator(model)
    rec: received signal
    noise: noise signal
    """
    rec = torch.reshape(rec, shape=(-1, 1))  # reshapes to column vector
    noise = torch.reshape(noise, shape=(-1, 1))  # reshapes to column vector
    rec_sample1, rec_sample2 = torch.split(
        rec, int(rec.shape[0] / 2), dim=0
    )  # splits rec into two havles
    noise_sample1, noise_sample2 = torch.split(
        noise, int(noise.shape[0] / 2), dim=0
    )  # splits noise into two halves
    joint = torch.cat(
        (rec_sample1, noise_sample1), 1
    )  # creates joint by pairing the first half of rec and noise
    marg = torch.cat(
        (rec_sample1, noise_sample2), 1
    )  # creates marginal by pairing the first half of rec and the second half of noise
    return joint, marg
