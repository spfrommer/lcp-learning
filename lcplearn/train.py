import sys
sys.path.append('..')

from argparse import ArgumentParser
import pdb

import torch
from torch.utils.data import TensorDataset, DataLoader

import sims

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

EPOCHS = 800
BATCH_SIZE = 32

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='out/data.npy')
    parser.add_argument('--modelpath', default='out/model.pt')
    parser.add_argument('--learntype', default='pytorch',
                        choices=['pytorch', 'custom'])
    parser.add_argument('modeltype', type=sims.ModelType,
                                     choices=list(sims.ModelType))
    opts = parser.parse_args()
    
    if opts.learntype == 'pytorch':
        model = sims.model_module(opts.modeltype)
        net, loss_func, optimizer, scheduler = model.learning_setup()
        net = net.to(device)

        states, ys, _ = model.load_data(opts.datapath)
        states = states.to(device)
        ys = ys.to(device)

        dataset = TensorDataset(states, ys)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        print("Using device: {}".format(device))
        print("Dataset size: {}".format(len(dataset)))
        print("Batch size: {}".format(BATCH_SIZE))

        train_loss = loss_func(net(states), ys, states).data.item()
        print('Starting loss: {}'.format(train_loss))

        for epoch in range(EPOCHS):
            if not scheduler is None:
                scheduler.step()

            for batch in dataloader:
                states_batch, ys_batch = batch
                
                ys_pred = net(states_batch)
                loss = loss_func(ys_pred, ys_batch, states_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = loss_func(net(states), ys, states).data.item()
            print('Finished epoch {} with loss: {}'.format(epoch, train_loss))

            torch.save(net.state_dict(), opts.modelpath)

        train_loss = loss_func(net(states), ys, states).data.item()
        print('Finished training with loss: {}'.format(train_loss))
        torch.save(net.state_dict(), opts.modelpath)
    elif opts.learntype == 'custom':
        print('No custom support')

if __name__ == "__main__": main()
