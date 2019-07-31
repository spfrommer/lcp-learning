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

import matplotlib.pyplot as plt
#import mplcursors

import pdb

soft_nonnegativity = False
soft_lambdas = False

def physics_variables(mu, data):
    n = data.shape[0]
    vs = torch.unsqueeze(data[:, 0], 1)
    vnexts = torch.unsqueeze(data[:, 1], 1)
    us = torch.unsqueeze(data[:, 2], 1)

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

    a1 = 1
    a2 = 1
    a3 = 5
    a4 = 5

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
    
    return {'cost': sum(costs), 'lambdaPlus': z[0,0],
            'lambdaMinus': z[0, 1], 'gamma': z[0, 2],
            'slack[0]': lcp_slack[0,0], 'slack[1]': lcp_slack[0, 1],
            'slack[2]': lcp_slack[0, 2]}

# Previous vel, next vel, u
data = torch.tensor([[1.0, 2.0, 2.0]])

mus = np.linspace(0.1, 10.0, num=100)
variables = []

for mu in mus:
    variables.append(physics_variables(mu, data))

# Convert list of dicts to dict of lists
variables = {k: [dic[k] for dic in variables] for k in variables[0]}

for var in variables:
    outs = [x.item() for x in variables[var]]
    plt.plot(mus, outs, linewidth=4)

plt.legend(variables.keys())
plt.grid()
plt.xlabel('mu')

#mplcursors.cursor(hover=True)

plt.show()
