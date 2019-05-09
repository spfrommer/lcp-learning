import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pdb

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from dynamics import SimType
import plot

epochs = 200

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', default='out/data.npy')
    parser.add_argument('simtype', type=SimType, choices=list(SimType))
    opts = parser.parse_args()

    states, lambdas = load_data(opts.path)

    net = Net(n_feature=states.size(1), n_hidden=5, n_output=lambdas.size(1))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(epochs):
        lambdas_pred = net(states)
        loss = loss_func(lambdas_pred, lambdas)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.data.numpy()

    print('Finished training with loss: {}'.format(loss_val.item()))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot.plot_data(ax, opts.path, opts.simtype)
    mins = states.min(dim=0).values
    maxs = states.max(dim=0).values
    plot.plot_net(ax, net, [float(mins[0]), float(maxs[0])],
                           [float(mins[1]), float(maxs[1])], opts.simtype)
    plt.show()

def load_data(path):
    data = np.load(path)
    states = torch.from_numpy(data[:, 0:2])
    lambdas = torch.from_numpy(data[:, 2])

    states, lambdas = Variable(states), Variable(lambdas)
    lambdas = lambdas.view(-1, 1)
    return states, lambdas

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == "__main__": main()
