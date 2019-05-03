import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np


data = np.load('out/data.npy')
x = torch.from_numpy(data[:, 0:1])
y = torch.from_numpy(data[:, 2])

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# this is one way to define a network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

epochs = 200

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# train the network
for t in range(epochs):
    print('Training epoch: {}'.format(t))
    print('Predicting')
    prediction = net(x)     # input x and predict based on x

    print('Calculating loss')
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    print('Backpropping')
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients

    print('Applying gradients')
    optimizer.step()        # apply gradients
    
    loss_val = loss.data.numpy()
    print(loss_val)
