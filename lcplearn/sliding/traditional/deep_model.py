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
    model = DeepNet(40)
    loss = structured_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    [90, 150, 250], gamma=0.3)

    return model, loss, optimizer, None

def structured_loss(net_out, next_xdots, states, net):
    lcp_slack = net_out
    lambdas = states[:, 2:5]
    
    #comp_term = torch.norm(lambdas * lcp_slack, 2)
    comp_term = torch.norm(torch.bmm(lambdas.unsqueeze(1),
                          torch.clamp(lcp_slack, min=0).unsqueeze(2)))
    nonneg_term = torch.norm(torch.clamp(-lcp_slack, min=0), 2)
    
    loss = (1 * comp_term + 1 * nonneg_term) / net_out.shape[0]
    return loss

class DeepNet(torch.nn.Module):
    def __init__(self, hidden_size):
        super(DeepNet, self).__init__()

        self.f1 = torch.nn.Linear(2, hidden_size)
        self.f2 = torch.nn.Linear(hidden_size, hidden_size)
        self.f3 = torch.nn.Linear(hidden_size, 3)

        self.G1 = torch.nn.Linear(2, hidden_size)
        self.G2 = torch.nn.Linear(hidden_size, hidden_size)
        self.G3 = torch.nn.Linear(hidden_size, 9)

    def forward(self, states):
        lambdas = states[:, 2:5]
        xus = states[:, 0:2]

        fxu = self.f3(F.relu(self.f2(F.relu(self.f1(xus)))))
        Gxu = self.G3(F.relu(self.G2(F.relu(self.G1(xus))))).view(-1, 3, 3)
        
        lcp_slack = fxu + torch.bmm(Gxu,lambdas.unsqueeze(2)).squeeze(2)
        
        return lcp_slack

if __name__ == "__main__": main()
