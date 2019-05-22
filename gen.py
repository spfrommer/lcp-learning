import logging
import numpy as np
import numpy.random as rand
from argparse import ArgumentParser
import pdb

from dynamics import *
from sims import *

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

def main():
    parser = ArgumentParser()
    parser.add_argument('simtype', type=SimType, choices=list(SimType))
    parser.add_argument('--path', default='out/data.npy')
    opts = parser.parse_args()

    run_rows = time_steps - 1
    dataset = np.zeros((runs * run_rows, variable_count(opts.simtype)))
    
    print('Generating {} runs...'.format(runs))
    bar = progressbar.ProgressBar(maxval=runs, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i in range(runs):
        bar.update(i+1)
        if opts.simtype == SimType.FALLING:
            p = random_falling_params()
            sol = falling_box_sim(p)
            data = marshal_falling_data(sol)
        elif opts.simtype == SimType.SLIDING:
            p = random_sliding_params()
            sol = sliding_box_sim(p)
            sol = process_sliding_solution(sol, p)
            data = marshal_sliding_data(sol)

        dataset[i * run_rows : (i+1) * run_rows, :] = data

    bar.finish()
    
    save_marshalled_data(dataset, opts.path)

if __name__ == "__main__": main()
