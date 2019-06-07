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
    model = BasisNet(False)
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

    loss = 1 * comp_term + 1 * nonneg_term 
    
    Gxu, _ = net.get_lcps(states)

    constraints = [(1, Gxu[:, 0, 0]), (-1, Gxu[:, 0, 1]),
                   (-1, Gxu[:, 1, 0]), (1, Gxu[:, 1, 1]),
                   (-1, Gxu[:, 2, 0])]
    for c in constraints:
        loss = loss + 50 * torch.norm(c[0] - c[1], 2)

    return loss

class BasisNet(torch.nn.Module):
    def __init__(self, include_G_weights):
        super(BasisNet, self).__init__()

        self.include_G_weights = include_G_weights

        self.psd_vec = torch.nn.Linear(2, 3, bias=False)
        self.psd_vec_bias = Parameter(torch.ones(3))
        self.nonneg_matrix = torch.nn.Linear(2, 9, bias=False)
        self.nonneg_matrix_bias = Parameter(torch.ones(9))
        self.antisym = torch.nn.Linear(2, 3, bias=False)
        self.antisym_bias = Parameter(torch.ones(3))

        self.antisym_basis = torch.tensor([[[0, 1, 0],
                                           [-1, 0, 0],
                                            [0, 0, 0]],
                                            [[0, 0, 1],
                                            [0, 0, 0],
                                           [-1, 0, 0]],
                                           [[0, 0, 0],
                                            [0, 0, 1],
                                            [0, -1, 0]]]).float()

        self.f = torch.nn.Linear(2, 3, bias=True)

        if not include_G_weights:
            self.psd_vec.weight.data.fill_(0)
            self.nonneg_matrix.weight.data.fill_(0)
            self.antisym.weight.data.fill_(0)

    def forward(self, states):
        lambdas = states[:, 2:5]
        Gxu, fxu = self.get_lcps(states)
        
        lcp_slack = fxu + torch.bmm(Gxu,lambdas.unsqueeze(2)).squeeze(2)
        
        return lcp_slack

    def get_lcps(self, states):
        lambdas = states[:, 2:5]
        xus = states[:, 0:2]
        
        fxu = self.f(xus)
        
        if self.include_G_weights:
            psd_vec = self.psd_vec(xus) + self.psd_vec_bias
            nonneg_matrix = self.nonneg_matrix(xus) + self.nonneg_matrix_bias
            antisym_vec = self.antisym(xus) + self.antisym_bias
        else:
            psd_vec = self.psd_vec_bias.expand(states.shape[0], 3)
            nonneg_matrix = self.nonneg_matrix_bias.expand(states.shape[0], 9)
            antisym_vec = self.antisym_bias.expand(states.shape[0], 3)

        psd_term = torch.bmm(psd_vec.unsqueeze(2), psd_vec.unsqueeze(1))
        
        nonneg_term = F.relu(nonneg_matrix.view(-1, 3, 3))
        
        antisym_vec = antisym_vec.unsqueeze(2).unsqueeze(3)
        antisym_basis = self.antisym_basis.expand(antisym_vec.shape[0], 3, 3, 3)
        antisym_term = torch.sum(antisym_vec * antisym_basis, dim=1)
        
        Gxu = psd_term + nonneg_term + antisym_term

        return Gxu, fxu

if __name__ == "__main__": main()
