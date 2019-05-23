import sys
sys.path.append('..')

from argparse import ArgumentParser
import pdb

import torch
from torch.utils.data import TensorDataset, DataLoader

import sims

EPOCHS = 50
BATCH_SIZE = 10000

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='out/data.npy')
    parser.add_argument('--modelpath', default='out/model.pt')
    parser.add_argument('modeltype', type=sims.ModelType,
                                     choices=list(sims.ModelType))
    opts = parser.parse_args()

    model = sims.model_module(opts.modeltype)
    net, loss_func, optimizer = model.learning_setup()

    states, ys, _ = model.load_data(opts.datapath)
    dataset = TensorDataset(states, ys)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for batch in dataloader:
            states_batch, ys_batch = batch
            
            ys_pred = net(states_batch)
            loss = loss_func(ys_pred, ys_batch, states_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_loss = loss_func(net(states), ys, states).data.numpy()

    print('Finished training with loss: {}'.format(train_loss.item()))
    torch.save(net.state_dict(), opts.modelpath)

if __name__ == "__main__": main()
