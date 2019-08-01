import torch
import torch.nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import numpy as np

import forwards

import pdb

soft_nonnegativity = False
soft_lambdas = False
force_max_dissipation_qp = False
force_max_dissipation_ext = False

forwards_fn = forwards.get_fowards_function(
        soft_nonnegativity, soft_lambdas,
        force_max_dissipation_qp, force_max_dissipation_ext) 

class Parameters:
    def __init__(self, mu):
        self.mu = mu

physics_variables = forwards.get_fowards_function(
        soft_nonnegativity, soft_lambdas,
        force_max_dissipation_qp, force_max_dissipation_ext) 

class PhysicsNet(torch.nn.Module):
    def __init__(self, startmu):
        super().__init__()
        self.mu = Parameter(torch.tensor([startmu]))

    def forward(self, data):
        cost, _ = forwards_fn(self, data)
        return cost

# Previous vel, next vel, u
data = torch.tensor([[1.0, 2.0, 2.0]])

evolutions = []
#for startmu in np.linspace(0.1, 10, num=30):
for startmu in [10.5]:
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
