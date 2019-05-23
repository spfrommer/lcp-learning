import numpy as np
from argparse import ArgumentParser
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import torch

from sims import *

MAX_POINT_RENDER = 3000
DATA_MARKER_COLOR = 'blue'
DATA_MARKER_SIZE = 3
PRED_MARKER_COLOR = 'red'
PRED_MARKER_SIZE = 3

def scatter_data(ax, xs, ys, zs):
    ax.scatter(xs[:MAX_POINT_RENDER],
               ys[:MAX_POINT_RENDER],
               zs[:MAX_POINT_RENDER],
               c=DATA_MARKER_COLOR, s=DATA_MARKER_SIZE)

def plot_data_old(ax, path, simtype):
    if simtype == SimType.FALLING:
        data = load_falling_data(path)
        ax.scatter(data.xs[:MAX_POINT_RENDER],
                   data.xdots[:MAX_POINT_RENDER],
                   data.lambdas[:MAX_POINT_RENDER],
                   c=DATA_MARKER_COLOR, s=DATA_MARKER_SIZE)
    elif simtype == SimType.SLIDING:
        data = load_sliding_data(path)
        ax.scatter(data.xdots[:MAX_POINT_RENDER],
                   data.us[:MAX_POINT_RENDER],
                   data.next_xdots[:MAX_POINT_RENDER],
                   c=DATA_MARKER_COLOR, s=DATA_MARKER_SIZE)

def surf_net(ax, net, xrange, yrange, net_process=lambda x:x):
    xs = torch.tensor(np.linspace(xrange[0], xrange[1], 30))
    ys = torch.tensor(np.linspace(yrange[0], yrange[1], 30))

    grid_x, grid_y = torch.meshgrid(xs, ys)
    xs_vec = grid_x.reshape(-1, 1).float()
    ys_vec = grid_y.reshape(-1, 1).float()

    zs = net_process(net(torch.cat((xs_vec, ys_vec), dim=1)))
    grid_z = zs.reshape(grid_x.shape)

    ax.plot_surface(grid_x.detach().numpy(),
                    grid_y.detach().numpy(),
                    grid_z.detach().numpy(),
                    cmap = cm.hot, alpha=0.2)

def scatter_net(ax, net, states, net_process=lambda x:x):
    zs = net_process(net(states))
    ax.scatter(states[:MAX_POINT_RENDER, 0].detach().numpy(),
               states[:MAX_POINT_RENDER, 1].detach().numpy(),
               zs[:MAX_POINT_RENDER].detach().numpy(),
               c=PRED_MARKER_COLOR, s=PRED_MARKER_SIZE)
    #ax.plot_trisurf(states[:MAX_POINT_RENDER, 0].detach().numpy(),
    #           states[:MAX_POINT_RENDER, 1].detach().numpy(),
    #           zs[:MAX_POINT_RENDER].detach().numpy())

def add_labels(ax, xlabel, ylabel, zlabel):
    ax.set_xlabel('x(k)')
    ax.set_ylabel('xdot(k)')
    ax.set_zlabel('lambda(k+1)')

def add_labels_old(ax, simtype):
    if simtype == SimType.FALLING:
        ax.set_xlabel('x(k)')
        ax.set_ylabel('xdot(k)')
        ax.set_zlabel('lambda(k+1)')
    elif simtype == SimType.SLIDING:
        ax.set_xlabel('xdot(k)')
        ax.set_ylabel('u(k+1)')
        ax.set_zlabel('xdot(k+1)')

if __name__ == "__main__": main()
