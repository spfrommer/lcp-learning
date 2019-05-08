import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import plot
epochs = 200

def main():
    states, lambdas = load_data('out/data.npy')

    net = Net(n_feature=states.size(1), n_hidden=3, n_output=lambdas.size(1))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
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
    plot.plot_data(ax, 'out/data.npy')
    plot.plot_net(ax, net, [0, 30], [-10, 10])
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
