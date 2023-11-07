# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
        )
        
        self.tail = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, out_size)
        )
    
    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

        inp = x.view(len(x), 3, 31, 31)
        inp = inp * (1.0 / 255)
        inp = (inp - 0.5) / 0.25
        out = self.main(inp)
        out = out.mean(-1).mean(-1)
        out = self.tail(out)

        return out

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        with torch.no_grad():
            preds = self.forward(x)
            loss = self.loss_fn(preds, y).item()
            return loss

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    lr = 0.001
    net = NeuralNet(lr, nn.CrossEntropyLoss(), 3, 4)
    #net.cuda()
    opt = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-3)
    data = get_dataset_from_arrays(train_set, train_labels)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    error = []
    for i in range(epochs):
        err = 0
        net.train()
        for k, batch in enumerate(loader):
            opt.zero_grad()
            # x, y = batch['features'].cuda(), batch['labels'].cuda()
            x, y = batch['features'], batch['labels']

            preds = net(x)
            loss = net.loss_fn(preds, y)
            loss.backward()
            opt.step()
            err += loss.item()
        error.append(err / (k + 1))
        #print("Epoch {}: {:.4f}".format(i + 1, error[-1]))

    net.cpu()
    net.eval()
    with torch.no_grad():
        preds = net(dev_set)
        y_hat = torch.argmax(preds, -1)
        y_hat = y_hat.cpu().numpy()

    return error, y_hat, net
