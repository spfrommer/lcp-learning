import matplotlib.pyplot as plt

from lcplearn import vis_utils
from lcplearn import lcp_utils

from . import lcp_structured_model as model
from . import dynamics

def analyze(opts):
    handle_vis(opts)

def handle_vis(opts):
    states, ys, data = model.load_data(opts.datapath)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    vis_utils.scatter_data(ax, data.xdots, data.us, data.next_xdots)

    vis_utils.add_labels(ax, 'xdot(k)', 'u(k+1)', 'xdot(k+1)')

    plt.show()
