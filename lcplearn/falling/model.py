import numpy as np
from argparse import ArgumentParser
import pdb

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from . import dynamics

epochs = 50

def load_data(path):
    data = dynamics.unmarshal_data(np.load(path)) 
    states = torch.from_numpy(np.vstack((
                data.xs, data.xdots)).transpose())
    ys = torch.from_numpy(data.lambdas)
    ys = ys.view(-1, 1)

    states, ys = Variable(states), Variable(ys)
    return states, ys, data

def learning_setup():
    model = StandardNet(n_feature=2,
                        n_hidden=5, n_output=1)
    # Add dummy arg to match function signature
    loss = lambda ys_p,ys,states: torch.nn.MSELoss().forward(ys_p, ys)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    return model, loss, optimizer

class StandardNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(StandardNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == "__main__": main()
