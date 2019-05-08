import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import torch

def main():
    plot_data('out/data.npy')

def plot_data(ax, path):
    data = np.load(path)


    point_count = 3000
    ax.scatter(data[:point_count, 0],
               data[:point_count, 1],
               data[:point_count, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('xdot')
    ax.set_zlabel('lambda')

def plot_net(ax, net, xrange, xdotrange):
    xs = torch.tensor(np.linspace(xrange[0], xrange[1], 30))
    xdots = torch.tensor(np.linspace(xdotrange[0], xdotrange[1], 30))

    grid_x, grid_xdot = torch.meshgrid(xs, xdots)
    xs_vec = grid_x.reshape(-1, 1).float()
    xdots_vec = grid_xdot.reshape(-1, 1).float()

    lambdas = net(torch.cat((xs_vec, xdots_vec), dim=1))
    grid_lambdas = lambdas.reshape(grid_x.shape)

    ax.plot_surface(grid_x.detach().numpy(),
                    grid_xdot.detach().numpy(),
                    grid_lambdas.detach().numpy(),
                    cmap = cm.hot, alpha=0.2)

    ax.set_xlabel('x')
    ax.set_ylabel('xdot')
    ax.set_zlabel('lambda')

    plt.show()

if __name__ == "__main__": main()
