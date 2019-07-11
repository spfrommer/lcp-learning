import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

def loss_fn(outputs, targets):
    l1loss = torch.norm(outputs - targets, 1)
    l2loss = torch.norm(outputs - targets, 2)
    
    # This works as expected
    # loss = 1 * l1loss + 1 * l2loss
    # Loss never changes, no matter what combination of
    # requires_grad I set
    #loss = torch.dot(torch.tensor([1.0, 1.0], requires_grad=False),
    #        torch.tensor([l1loss, l2loss], requires_grad=True))
    # Correct version:
    loss = torch.dot(torch.tensor([1.0, 1.0]),
            torch.cat((l1loss.unsqueeze(0), l2loss.unsqueeze(0))))
    return loss

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
