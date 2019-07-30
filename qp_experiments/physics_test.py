import torch
import torch.nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction
from qpth.qp import QPSolvers

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

import pdb

soft_lambdas = False
soft_nonnegativity = False

class PhysicsNet(torch.nn.Module):
    def __init__(self, startmu):
        super().__init__()
        self.mu = Parameter(torch.tensor([startmu]))

    def forward(self, data):
        n = data.shape[0]
        vs = torch.unsqueeze(data[:, 0], 1)
        vnexts = torch.unsqueeze(data[:, 1], 1)
        us = torch.unsqueeze(data[:, 2], 1)

        mu = self.mu

        beta = vnexts - vs - us
        
        pad_base = torch.zeros(6)

        G = torch.tensor([[1.0, -1, 1], [-1, 1, 1], [-1, -1, 0]]).repeat(n, 1, 1)
        
        Gpad = torch.tensor([[1.0, -1, 1, 0, 0, 0, 0, 0, 0],
                             [-1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [-1, -1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(n, 1, 1)
        
        fmats = torch.tensor([[1.0, 1, 0], 
                              [-1, -1, 0],
                              [0, 0, 1]]).unsqueeze(0).repeat(n, 1, 1)
        fvecs = torch.cat((vs, us, mu * torch.ones(vs.shape)), 1).unsqueeze(2)
        f = torch.bmm(fmats, fvecs)
        batch_zeros = torch.zeros(f.shape)
        fpad = torch.cat((f, batch_zeros, batch_zeros), 1)
        
        # For prediction error
        A = torch.tensor([[1.0, -1, 0, 0, 0, 0, 0, 0, 0],
                          [-1, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(n, 1, 1)
        beta_zeros = torch.zeros(beta.shape)
        b = torch.cat((-2 * beta, 2 * beta, beta_zeros, beta_zeros, beta_zeros, 
                       beta_zeros, beta_zeros, beta_zeros, beta_zeros), 1).unsqueeze(2)
        
        inequality_slack_penalty = torch.tensor([0.0, 0, 0, 1, 1, 1, 0, 0, 0]).repeat(n, 1).unsqueeze(2)
        lambdas_slack_penalty = torch.tensor([0.0, 0, 0, 0, 0, 0, 1, 1, 1]).repeat(n, 1).unsqueeze(2)

        a1 = 10
        a2 = 1
        a3 = 100
        a4 = 100

        Q = 2 * a1 * A + 2 * a2 * Gpad
        p = a1 * b + a2 * fpad + a3 * inequality_slack_penalty + a4 * lambdas_slack_penalty
        
        # Constrain slacks (but not lambdas) to be >= 0
        R = torch.cat((torch.zeros(6, 3), -torch.eye(6)), 1).repeat(n, 1, 1)
        h = torch.zeros((1, 6)).repeat(n, 1).unsqueeze(2)

        I = torch.eye(3).repeat(n, 1, 1)

        if soft_nonnegativity:
            # Constrain G lambda + s(1:3) >= 0
            R = torch.cat((R, -torch.cat((G, I, torch.zeros(n, 3, 3)), 2)), 1)
        else:
            # Constrain G lambda + f >= 0
            R = torch.cat((R, torch.cat((-G, torch.zeros(n, 3, 6)), 2)), 1)

        # This is the same with soft or hard nonnegativity constraint
        h = torch.cat((h, f), 1)
        
        if soft_lambdas:
            # Constrain lambda + s(4:7) >= 0
            R = torch.cat((R, -torch.cat((I, torch.zeros(n, 3, 3), I), 2)), 1)
            h = torch.cat((h, torch.zeros(n, 3, 1)), 1)
        else:
            # Constrain lambda >= 0
            R = torch.cat((R, torch.cat((-I, torch.zeros(n, 3, 6)), 2)), 1)
            h = torch.cat((h, torch.zeros(n, 3, 1)), 1)

        Qmod = 0.5 * (Q + Q.transpose(1, 2)) + 0.001 * torch.eye(9).repeat(n, 1, 1)
        
        z = QPFunction(check_Q_spd=False)(Qmod, p.squeeze(2), R, h.squeeze(2), torch.tensor([]), torch.tensor([]))

        lcp_slack = torch.bmm(Gpad, z.unsqueeze(2)) + fpad

        costs = 0.5 * torch.bmm(z.unsqueeze(1), torch.bmm(Qmod, z.unsqueeze(2))) \
                + torch.bmm(p.transpose(1, 2), z.unsqueeze(2)) + a1 * beta**2
        
        return sum(costs)

# Previous vel, next vel, u
data = torch.tensor([[1.0, 2.0, 2.0], [2.0, 3.0, 2.0]])

evolutions = []
#for startmu in np.linspace(0.1, 10, num=30):
for startmu in [30.0]:
    net = PhysicsNet(startmu)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    evolution = []
    for epoch in range(10000):
        # Zero the gradients
        optimizer.zero_grad()

        error = net(data)

        loss = torch.norm(error, 2)
        evolution.append([epoch, net.mu.item()])
        print('epoch: {}, loss: {:0.4f}, mu: {:0.4f}'.format(
            epoch, loss.item(), net.mu.item()))
        
        # perform a backward pass (backpropagation)
        loss.backward()
        
        # Update the parameters
        # optimizer.step()
        for p in net.parameters():
            if p.requires_grad:
                p.data.add_(0.1, -p.grad.data)
                
    evolutions.append(evolution)

evolutions_array = np.array(evolutions)
np.save('soft_evolution', evolutions_array)
