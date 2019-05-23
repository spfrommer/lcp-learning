import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pdb

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sims import *
import plot

epochs = 50

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', default='out/data.npy')
    parser.add_argument('simtype', type=SimType, choices=list(SimType))
    opts = parser.parse_args()

    net, loss_func, optimizer = setup_net(opts.simtype)

    states, ys = load_data(opts.path, opts.simtype)
    dataset = TensorDataset(states, ys)
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            states_batch, ys_batch = batch
            
            ys_pred = net(states_batch)
            loss = loss_func(ys_pred, ys_batch, states_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_loss = loss_func(net(states), ys, states).data.numpy()

    print('Finished training with loss: {}'.format(train_loss.item()))
    handle_print(states, net, opts)
    handle_plot(states, net, opts)
    
def handle_plot(states, net, opts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot.plot_data(ax, opts.path, opts.simtype)
    if opts.simtype == SimType.FALLING:
        mins = states.min(dim=0).values
        maxs = states.max(dim=0).values
        plot.surf_net(ax, net,
                        [float(mins[0]), float(maxs[0])],
                        [float(mins[1]), float(maxs[1])])
    elif opts.simtype == SimType.SLIDING:
        plot.data_net(ax, net, states,
                net_process=lambda out: join_sign(out, pos_first=True))

    plot.add_labels(ax, opts.simtype)
    plt.show()

def handle_print(states, net, opts):
    if opts.simtype == SimType.SLIDING:
        print('f:')
        print(net.f.weight)
        print(net.f.bias)
        print('G:')
        print(net.G.weight)
        print(net.G.bias)

def load_data(path, simtype):
    if simtype == SimType.FALLING:
        data = load_falling_data(path)
        states = torch.from_numpy(np.vstack((
                    data.xs, data.xdots)).transpose())
        ys = torch.from_numpy(data.lambdas)
        ys = ys.view(-1, 1)
    elif simtype == SimType.SLIDING:
        data = load_sliding_data(path)
        states = torch.from_numpy(np.vstack((
                    data.xdots, data.us, data.lambdas)).transpose())
        ys = split_sign(torch.from_numpy(data.next_xdots),
                        pos_first=True)

    states, ys = Variable(states), Variable(ys)
    return states, ys

def setup_net(simtype):
    if simtype == SimType.FALLING:
        net = StandardNet(n_feature=2,
                          n_hidden=5, n_output=1)
        # Add dummy arg to match function signature
        loss = lambda ys_p,ys,states: torch.nn.MSELoss().forward(ys_p, ys)
    elif simtype == SimType.SLIDING:
        net = StructuredNet()
        #loss = lambda ys_p,ys,states: torch.nn.MSELoss().forward(ys_p, ys)
        loss = structured_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=0.2)
    return net, loss, optimizer

# Turns a n-dim vector into an nx2 matrix
# If pos_first, first column is all positive values (negs set to 0)
# Second col is abs value of all negative values (pos set to 0)
def split_sign(tensor, pos_first):
    tensor_pos = tensor.clone()
    tensor_neg = tensor.clone()

    tensor_pos[tensor_pos < 0] = 0
    tensor_neg[tensor_neg > 0] = 0

    tensor_pos = tensor_pos.unsqueeze(1)
    tensor_neg = tensor_neg.unsqueeze(1) * (-1)
    if pos_first:
        tensor = torch.cat((tensor_pos, tensor_neg), 1)
    else:
        tensor = torch.cat((tensor_neg, tensor_pos), 1)
    return tensor

# Inverse of split_sign
def join_sign(tensor, pos_first):
    tensor = tensor.clone()
    if pos_first:
        tensor[:, 1] = tensor[:, 1] * -1
    else:
        tensor[:, 0] = tensor[:, 0] * -1
    
    return tensor.sum(1)

def mse_loss(ys_pred, ys, states):
    return torch.norm(ys_pred - ys, 2)

def structured_loss(next_xdots_pred, next_xdots, states):
    lambdas = split_sign(states[:, 2], pos_first=True)
    
    err_term = torch.norm(next_xdots_pred - next_xdots, 2)
    comp_term = torch.norm(next_xdots_pred * lambdas, 1)
    nonneg_term = torch.norm(torch.max(
        torch.zeros(next_xdots_pred.shape), -next_xdots_pred), 1)
    
    w = torch.tensor([1, 1, 0])

    return w[0] * err_term + w[1] * comp_term + w[2] * nonneg_term

class StructuredNet(torch.nn.Module):
    def __init__(self):
        super(StructuredNet, self).__init__()
        self.f = torch.nn.Linear(2, 2, bias=True)
        self.G = torch.nn.Linear(2, 4, bias=True)

    def forward(self, states):
        # An nx2 array with first col neg and second col pos components
        lambdas = split_sign(states[:, 2], pos_first=True)
        xus = states[:, 0:2]
        fxu = self.f(xus)
        Gxu = self.G(xus).view(-1, 2, 2)
        
        xdots = fxu + torch.bmm(Gxu, lambdas.unsqueeze(2)).squeeze(2) 

        #return xdots
        return F.relu(xdots)

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
