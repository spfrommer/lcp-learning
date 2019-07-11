import sys
sys.path.append('..')

from argparse import ArgumentParser
import pdb

import math

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import sims

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

EPOCHS = 10000
DATA_SIZE = None
BATCH_SIZE = 32

def get_data(dataset):
    states = (dataset.dataset.tensors[0])[dataset.indices, :]
    ys = (dataset.dataset.tensors[1])[dataset.indices]
    return states, ys

def get_losses(train_dataset, validation_dataset, net, loss_func):
    train_states, train_ys = get_data(train_dataset)
    # train_loss = loss_func(net(train_states), train_ys,
            # train_states, net).data.item()
    train_loss = torch.norm(loss_func(net(train_states), train_ys,
            train_states, net), 1)
    
    validation_states, validation_ys = get_data(validation_dataset)
    validation_loss = torch.norm(loss_func(net(validation_states), validation_ys,
            validation_states, net), 1)

    return train_loss, validation_loss


def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='out/data.npy')
    parser.add_argument('--modelpath', default='out/model.pt')
    parser.add_argument('modeltype', type=sims.ModelType,
                                     choices=list(sims.ModelType))
    opts = parser.parse_args()
    
    model = sims.model_module(opts.modeltype)
    net, loss_func, optimizer, scheduler = model.learning_setup()
    net = net.to(device)

    states, ys, _ = model.load_data(opts.datapath)
    states = states[0:min(states.shape[0], DATA_SIZE), :]
    ys = ys[0:min(states.shape[0], DATA_SIZE)]
    states = states.to(device)
    ys = ys.to(device)

    dataset = TensorDataset(states, ys)
    train_split = int(len(dataset) * 0.8)
    train_dataset, validation_dataset = random_split(dataset, 
            [train_split, len(dataset) - train_split])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Using device: {}".format(device))
    print("Dataset size: {}".format(len(dataset)))
    print("Batch size: {}".format(BATCH_SIZE))

    train_loss, validation_loss = get_losses(train_dataset,
                validation_dataset, net, loss_func)
    print('Starting train loss {:.2e} and validation loss {:.2e}'.format(
        train_loss, validation_loss))

    for epoch in range(EPOCHS):
        if not scheduler is None:
            scheduler.step()

        for batch in train_dataloader:
            states_batch, ys_batch = batch
            
            ys_pred = net(states_batch)
            loss = loss_func(ys_pred, ys_batch, states_batch, net)
            
            optimizer.zero_grad()
            loss.backward()
            # For vector valued loss 
            #loss.backward(torch.ones(loss.size()))
            optimizer.step()

        train_loss, validation_loss = get_losses(train_dataset,
                    validation_dataset, net, loss_func)
        print('Finished epoch {} with train loss {:.2e} and validation loss {:.2e}'.format(epoch, train_loss, validation_loss))

        torch.save(net.state_dict(), opts.modelpath)

    train_loss = loss_func(net(states), ys, states, net).data.item()
    print('Finished training with loss: {}'.format(train_loss))
    torch.save(net.state_dict(), opts.modelpath)

def get_back(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                get_back(n[0])

if __name__ == "__main__": main()
