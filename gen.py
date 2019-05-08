import logging
import numpy as np
import numpy.random as rand

import dynamics
from dynamics import SimParams

import progressbar

g = 1.0
dt = 1.0
time_steps = 30

runs = 1000

def random_falling_params():
    p = FallingSimParams(x0=rand.uniform(0,20),   xdot0=rand.uniform(-5,5),
                         lambda0=0.0,             g=g,
                         dt=dt,                   time_steps=time_steps)
    return p

# Lines up xs/xdots from one time stop to lambdas of the next
# And concatenates them into a matrix
# Goes from n entries -> n - 1
# Used to prep data for input into ML
def format_data(xs, xdots, lambdas):
    xs = xs[:-1]
    xdots = xdots[:-1]
    lambdas = lambdas[1:]
    return np.hstack((xs, xdots, lambdas))

def main():
    run_rows = time_steps - 1
    dataset = np.zeros((runs * run_rows, 3))
    
    print('Generating {} runs...'.format(runs))
    bar = progressbar.ProgressBar(maxval=runs, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i in range(runs):
        bar.update(i+1)
        p = random_falling_params()
        xs, xdots, lambdas = dynamics.falling_box_sim(p)
        data = format_data(xs, xdots, lambdas)
        dataset[i * run_rows : (i+1) * run_rows, :] = data

    bar.finish()

    np.save('./out/data.npy', dataset.astype(np.float32))

if __name__ == "__main__": main()
