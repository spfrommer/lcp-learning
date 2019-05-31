import matplotlib.pyplot as plt

from lcplearn import vis_utils
from lcplearn import lcp_utils

from . import model
from . import dynamics

def analyze(net, opts):
    handle_print(net)
    #handle_vis(net, opts)

def handle_vis(net, opts):
    states, ys, data = model.load_data(opts.datapath)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    vis_utils.scatter_data(ax, data.xdots, data.us, data.next_xdots)
    
    vis_utils.scatter_net(ax, net, states,  
        net_process=lambda out: lcp_utils.join_sign(out,pos_first=True))

    vis_utils.add_labels(ax, 'xdot(k)', 'u(k+1)', 'xdot(k+1)')

    plt.show()

def handle_print(net):
    print('f weights:')
    print(net.f.weight)
    print('f biases:')
    print(net.f.bias)
    print('G weights:')
    print(net.G.weight)
    print('G biases:')
    print(net.G.bias)
