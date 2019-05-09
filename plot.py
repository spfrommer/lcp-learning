import numpy as np
from argparse import ArgumentParser
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import torch

from dynamics import SimType

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', default='out/data.npy')
    parser.add_argument('simtype', type=SimType, choices=list(SimType))
    opts = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_data(ax, opts.path, opts.simtype)
    plt.show()

def add_labels(ax, simtype):
    if simtype == SimType.FALLING:
        ax.set_xlabel('x')
        ax.set_ylabel('xdot')
    elif simtype == SimType.SLIDING:
        ax.set_xlabel('u')
        ax.set_ylabel('xdot')
    ax.set_zlabel('lambda')


def plot_data(ax, path, simtype):
    data = np.load(path)

    point_count = 3000
    ax.scatter(data[:point_count, 0],
               data[:point_count, 1],
               data[:point_count, 2])
    add_labels(ax, simtype)

def plot_net(ax, net, xrange, yrange, simtype):
    xs = torch.tensor(np.linspace(xrange[0], xrange[1], 30))
    xdots = torch.tensor(np.linspace(yrange[0], yrange[1], 30))

    grid_x, grid_xdot = torch.meshgrid(xs, xdots)
    xs_vec = grid_x.reshape(-1, 1).float()
    xdots_vec = grid_xdot.reshape(-1, 1).float()

    lambdas = net(torch.cat((xs_vec, xdots_vec), dim=1))
    grid_lambdas = lambdas.reshape(grid_x.shape)

    ax.plot_surface(grid_x.detach().numpy(),
                    grid_xdot.detach().numpy(),
                    grid_lambdas.detach().numpy(),
                    cmap = cm.hot, alpha=0.2)
    add_labels(ax, simtype)

if __name__ == "__main__": main()
