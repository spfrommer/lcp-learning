import matplotlib.pyplot as plt

from lcplearn import vis_utils

from . import model
from . import dynamics

def analyze(net, opts):
    handle_vis(net, opts)

def handle_vis(net, opts):
    states, ys, data = model.load_data(opts.datapath)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    vis_utils.scatter_data(ax, data.xs, data.xdots, data.lambdas)

    mins = states.min(dim=0).values
    maxs = states.max(dim=0).values
    vis_utils.surf_net(ax, net,
                    [float(mins[0]), float(maxs[0])],
                    [float(mins[1]), float(maxs[1])])

    vis_utils.add_labels(ax, 'x(k)', 'xdot(k)', 'lambda(k+1)')

    plt.show()
