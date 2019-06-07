import numpy as np
from argparse import ArgumentParser
import pdb

import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

from lcplearn import lcp_utils
from . import dynamics
    
def load_data(path):
    data = dynamics.unmarshal_data(np.load(path)) 
    states = torch.from_numpy(np.vstack((
                data.xdots, data.us, data.poslambdas,
                data.neglambdas, data.gammas)).transpose())
    ys = torch.from_numpy(data.next_xdots)

    states, ys = Variable(states), Variable(ys)
    return states, ys, data

def learning_setup():
    model = LcpStructuredNet(False, False)
    loss = structured_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    [90, 150, 250], gamma=0.3)

    return model, loss, optimizer, scheduler

def structured_loss(net_out, next_xdots, states, net):
    lcp_slack = net_out
    lambdas = states[:, 2:5]

    comp_term = torch.norm(lambdas * lcp_slack, 2)
    nonneg_term = torch.norm(torch.clamp(-lcp_slack, min=0), 2)

    constraints = [(1, net.G_bias[0]), (1, net.G_bias[4]),
                   (-1, net.G_bias[6])]

    loss = 1 * comp_term + 1 * nonneg_term 
    for c in constraints:
        loss = loss + 50 * torch.norm(c[0] - c[1], 2)

    return loss

class LcpStructuredNet(torch.nn.Module):
    def __init__(self, warm_start, include_G_weights):
        super(LcpStructuredNet, self).__init__()
        self.include_G_weights = include_G_weights

        self.f = torch.nn.Linear(2, 3, bias=False)
        self.f_bias = torch.nn.Parameter(torch.ones(3))
        self.G = torch.nn.Linear(2, 9, bias=False)
        self.G_bias = torch.nn.Parameter(torch.ones(9))

        # Correct dynamics solution
        if warm_start:
            self.G.weight.data.fill_(0)
            self.G.weight.data = self.add_noise(self.G.weight.data)
            self.G_bias = Parameter(self.add_noise(
                    torch.tensor([1, -1, 1,
                                 -1,  1, 1,
                                 -1, -1, 0]).float()))

            self.f.weight = Parameter(self.add_noise(
                    torch.tensor([[1,  1],
                                 [-1, -1],
                                  [0,  0]]).float()))
            self.f_bias = Parameter(self.add_noise(
                    torch.tensor([0, 0, 1]).float()))
        else:
            torch.nn.init.xavier_uniform_(self.f.weight)
            torch.nn.init.xavier_uniform_(self.G.weight)

        if not include_G_weights:
            self.G.weight.data.fill_(0)
    
    def add_noise(self, tensor):
        m = torch.distributions.normal.Normal(0, 0.1)
        return tensor + m.sample(tensor.shape).float()

    def forward(self, states):
        lambdas = states[:, 2:5]
        xus = states[:, 0:2]
        fxu = self.f(xus) + self.f_bias
        
        if self.include_G_weights:
            Gxu = (self.G(xus) + self.G_bias).view(-1, 3, 3)
        else:
            Gxu = (self.G_bias.expand(states.shape[0], 9)).view(-1, 3, 3) 
        
        lcp_slack = fxu + torch.bmm(Gxu,lambdas.unsqueeze(2)).squeeze(2)
        
        return lcp_slack

if __name__ == "__main__": main()
