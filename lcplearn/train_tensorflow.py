import sys
sys.path.append('..')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from argparse import ArgumentParser
import pdb

import tensorflow as tf
import numpy as np

from sliding.traditional import dynamics

EPOCHS = 800
BATCH_SIZE = 1

def load_data(path):
    data = dynamics.unmarshal_data(np.load(path)) 
    states = np.vstack((data.xdots, data.us, data.poslambdas,
                        data.neglambdas, data.gammas))
    ys = data.next_xdots
    train_dataset = tf.data.Dataset.from_tensor_slices((states.T, ys))
    return states, ys, train_dataset

def structured_loss(net_out, next_xdots, states, net):
    # G_loss = tf.sqrt(tf.reduce_sum(tf.square(net.G.weights[0]))) + \
           # tf.sqrt(tf.reduce_sum(tf.square(net.G.weights[1]))) 
    # f_loss = tf.sqrt(tf.reduce_sum(tf.square(net.f.weights[0]))) + \
           # tf.sqrt(tf.reduce_sum(tf.square(net.f.weights[1]))) 

    # return G_loss + f_loss

    lcp_slack = net_out
    lambdas = states[:, 2:5]
    return tf.sqrt(tf.reduce_sum(tf.square(lcp_slack)))
    
    # #comp_term = torch.norm(lambdas * lcp_slack, 2)
    # comp_term = torch.norm(torch.bmm(lambdas.unsqueeze(1),
                          # torch.clamp(lcp_slack, min=0).unsqueeze(2)))
    # nonneg_term = torch.norm(torch.clamp(-lcp_slack, min=0), 2)

    # #constraints = [(1, net.G_bias[0]), (1, net.G_bias[4]),
    # #               (-1, net.G_bias[6])]
    # constraints = []

    # loss = 1 * comp_term + 1 * nonneg_term 
    # for c in constraints:
        # loss = loss + 50 * torch.norm(c[0] - c[1], 2)

    # return loss

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='out/data.npy')
    opts = parser.parse_args()

    states, ys, dataset = load_data(opts.datapath)
    print("Dataset size: {}".format(dataset._tensors[0].shape[0]))
    print("Batch size: {}".format(BATCH_SIZE))

    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
    
    net = LcpStructuredNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_func = structured_loss

    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataset):
            with tf.GradientTape() as tape:
                states_batch, ys_batch = batch
            
                ys_pred = net(states_batch)
                loss = loss_func(ys_pred, ys_batch, states_batch, net)
            
            grads = tape.gradient(loss, net.trainable_weights)
            optimizer.apply_gradients(zip(grads, net.trainable_weights))
        
        train_loss = loss_func(net(states), ys, states, net)
        print('Finished epoch {} with loss: {}'.format(epoch, train_loss))
        print(net.get_weights())

    train_loss = loss_func(net(states), ys, states, net).data.item()
    print('Finished training with loss: {}'.format(train_loss))
    torch.save(net.state_dict(), opts.modelpath)

class LcpStructuredNet(tf.keras.Model):
    def __init__(self):
        super(LcpStructuredNet, self).__init__()
        # self.f = tf.keras.layers.Dense(3,
                # kernel_initializer=tf.keras.initializers.glorot_normal())
        # self.G = tf.keras.layers.Dense(9,
                # kernel_initializer=tf.keras.initializers.glorot_normal())
        self.f = tf.keras.layers.Dense(3,
                weights=[np.random.uniform(size=(2,3)),
                         np.random.uniform(size=3)])
        self.G = tf.keras.layers.Dense(9,
                weights=[np.random.uniform(size=(2,9)),
                         np.random.uniform(size=9)])

    def call(self, states):
        lambdas = tf.reshape(states[:, 2:5], [-1, 3, 1])
        xus = states[:, 0:2]

        fxu = tf.reshape(self.f(xus), [-1, 3, 1])
        Gxu = tf.reshape(self.G(xus), [-1, 3, 3])

        lcp_slack = fxu + tf.matmul(Gxu, lambdas)
        return lcp_slack

if __name__ == "__main__": main()

