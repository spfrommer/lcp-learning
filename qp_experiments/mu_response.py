import torch
import torch.nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import mplcursors

import forwards

import pdb

soft_nonnegativity = False
soft_lambdas = False
l1_softness = False
force_max_dissipation_qp = False
force_max_dissipation_ext = False

class Parameters:
    def __init__(self, mu):
        self.mu = mu

physics_variables = forwards.get_fowards_function(
        soft_nonnegativity, soft_lambdas, l1_softness,
        force_max_dissipation_qp, force_max_dissipation_ext) 

# Previous vel, next vel, u
data = torch.tensor([[1.0, 2.0, 2.0]])

mus = np.linspace(0.1, 30.0, num=200)
variables = []

for mu in mus:
    params = Parameters(mu)
    _, vs = physics_variables(params, data) 
    variables.append(vs)

# Convert list of dicts to dict of lists
variables = {k: [dic[k] for dic in variables] for k in variables[0]}

for i, var in enumerate(variables):
    outs = [x.item() for x in variables[var]]
    plt.plot(mus, outs, linewidth = 4, alpha = 1)

plt.legend(variables.keys())
plt.grid()
plt.xlabel('mu')
plt.title('v_k=1, v_k+1=2, u=2')

#mplcursors.cursor(hover=True)

plt.show()
