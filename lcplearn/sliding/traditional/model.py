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
    model = StructuredNet()
    loss = structured_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    return model, loss, optimizer

def structured_loss(net_out, next_xdots, states):
    lcp_slack = net_out
    lambdas = states[:, 2:5]

    comp_term = torch.norm(lambdas * lcp_slack, 2)
    nonneg_term = torch.norm(torch.max(
        torch.zeros(lcp_slack.shape).double(), -lcp_slack), 2)
    magnitude_term = torch.norm(lcp_slack.reshape(1, -1).squeeze(), 2)
    
    w = torch.tensor([1, 0, 0])
    loss = w[0] * comp_term + w[1] * nonneg_term + w[2] * magnitude_term
    
    #if loss > 0: pdb.set_trace()
    return loss

class StructuredNet(torch.nn.Module):
    def __init__(self):
        super(StructuredNet, self).__init__()
        self.f = torch.nn.Linear(2, 3, bias=True).double()
        self.G = torch.nn.Linear(2, 9, bias=True).double()

        #torch.nn.init.xavier_uniform_(self.f.weight)
        #torch.nn.init.xavier_uniform_(self.G.weight)
        
        # Trivial solution
        #self.f.weight.data.fill_(0)
        #self.f.bias.data.fill_(0)
        #self.G.weight.data.fill_(0)
        #self.G.bias.data.fill_(0)

        # Correct dynamics solution
        self.G.weight.data.fill_(0)
        self.G.weight.data = self.add_noise(self.G.weight.data)
        self.G.bias = Parameter(self.add_noise(
                torch.tensor([1, -1, 1,
                             -1,  1, 1,
                             -1, -1, 0]).double()))
        self.f.weight = Parameter(self.add_noise(
                torch.tensor([[1,  1],
                             [-1, -1],
                              [0,  0]]).double()))
        # 1 = mu * m * g
        self.f.bias = Parameter(self.add_noise(
                torch.tensor([0, 0, 1]).double()))
    
    def add_noise(self, tensor):
        m = torch.distributions.normal.Normal(0, 0.1)
        return tensor + m.sample(tensor.shape).double()

    def forward(self, states):
        lambdas = states[:, 2:5]
        xus = states[:, 0:2]
        fxu = self.f(xus)
        Gxu = self.G(xus).view(-1, 3, 3)
        lcp_slack = fxu + torch.bmm(Gxu,lambdas.unsqueeze(2)).squeeze(2)
    
        return lcp_slack

if __name__ == "__main__": main()
