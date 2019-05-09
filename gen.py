import logging
import numpy as np
import numpy.random as rand
from argparse import ArgumentParser
import pdb

from dynamics import *

import progressbar

g = 1.0
dt = 1.0
time_steps = 30

runs = 1000

def random_falling_params():
    p = FallingSimParams(x0=rand.uniform(0,20), xdot0=rand.uniform(-5,5),
                         lambda0=0.0,           g=g,
                         dt=dt,                 time_steps=time_steps)
    return p

def random_sliding_params(): 
    mu = 1;
    p = SlidingSimParams(x0=rand.uniform(-5,5), xdot0=rand.uniform(-5,5),
                         mu=mu,                 g=g,
                         dt=dt,                 time_steps=time_steps,
                         us=rand.uniform(-3,3,size=(time_steps, 1)))
    return p

# Lines up xs/xdots from one time step to lambdas of the next
# And concatenates them into a matrix
# Goes from n entries -> n - 1
# Used to prep data for input into ML
def format_falling_data(xs, xdots, lambdas):
    xs = xs[:-1]
    xdots = xdots[:-1]
    lambdas = lambdas[1:]
    return np.hstack((xs, xdots, lambdas))

# Similar idea but for sliding data
# This time we use us from the next time step
def format_sliding_data(us, xdots, lambdas):
    us = us[1:]
    xdots = xdots[:-1]
    lambdas = lambdas[1:]
    return np.hstack((us, xdots, lambdas))

def main():
    parser = ArgumentParser()
    parser.add_argument('simtype', type=SimType, choices=list(SimType))
    parser.add_argument('--path', default='out/data.npy')
    opts = parser.parse_args()

    run_rows = time_steps - 1
    dataset = np.zeros((runs * run_rows, 3))
    
    print('Generating {} runs...'.format(runs))
    bar = progressbar.ProgressBar(maxval=runs, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i in range(runs):
        bar.update(i+1)
        if opts.simtype == SimType.FALLING:
            p = random_falling_params()
            sol = falling_box_sim(p)
            data = format_falling_data(sol.xs, sol.xdots, sol.lambdas)
        elif opts.simtype == SimType.SLIDING:
            p = random_sliding_params()
            sol = sliding_box_sim(p)
            sol = process_sliding_solution(sol, p)
            data = format_sliding_data(p.us, sol.xdots, sol.lambdas)

        dataset[i * run_rows : (i+1) * run_rows, :] = data

    bar.finish()

    np.save(opts.path, dataset.astype(np.float32))

if __name__ == "__main__": main()
