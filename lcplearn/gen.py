import sys
sys.path.append('..')

import logging
import numpy as np
import numpy.random as rand
from argparse import ArgumentParser
import pdb

import progressbar

import falling.dynamics as FallingDynamics
import sliding.direct.dynamics as SlidingDirectDynamics
import sliding.traditional.dynamics as SlidingTraditionalDynamics
import sims

g = 1.0
dt = 1.0
time_steps = 30

runs = 1000

def random_falling_params():
    pp = FallingDynamics.PhysicsParams(g=g, dt=dt)
    sp = FallingDynamics.SimParams(
                        x0=rand.uniform(0,20), xdot0=rand.uniform(-5,5),
                        lambda0=0.0,           time_steps=time_steps)
    return pp, sp

def random_sliding_direct_params(): 
    pp = SlidingDirectDynamics.PhysicsParams(
        us=rand.uniform(-3,3,size=(time_steps, 1)), g=g, mu=1, dt=dt)
    sp = SlidingDirectDynamics.SimParams(time_steps=time_steps,
        x0=rand.uniform(-5,5), xdot0=rand.uniform(-5,5))
    return pp, sp

def random_sliding_traditional_params(): 
    pp = SlidingTraditionalDynamics.PhysicsParams(
        us=rand.uniform(-3,3,size=(time_steps, 1)), g=g, mu=1)
    sp = SlidingTraditionalDynamics.SimParams(time_steps=time_steps,
        x0=rand.uniform(-3,3), xdot0=rand.uniform(-3,3))
    return pp, sp

def main():
    parser = ArgumentParser()
    parser.add_argument('simtype', type=sims.SimType,
                                   choices=list(sims.SimType))
    parser.add_argument('--path', default='out/data.npy')
    opts = parser.parse_args()
    
    dynamics = sims.dynamics_module(opts.simtype)

    run_rows = time_steps - 1
    dataset = np.zeros((runs * run_rows,
                        dynamics.MARSHALLED_SIZE))
    
    print('Generating {} runs...'.format(runs))
    bar = progressbar.ProgressBar(maxval=runs, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ',
                 progressbar.Percentage()])
    bar.start()

    for i in range(runs):
        bar.update(i+1)

        if opts.simtype == sims.SimType.FALLING:
            pp, sp = random_falling_params()
        elif opts.simtype == sims.SimType.SLIDING_DIRECT:
            pp, sp = random_sliding_direct_params()
        elif opts.simtype == sims.SimType.SLIDING_TRADITIONAL:
            pp, sp = random_sliding_traditional_params()

        sol = dynamics.sim(pp, sp)
        if dynamics.HAS_PROCESSING:
            sol = dynamics.process_solution(sol, pp)
        data = dynamics.marshal_data(sol)

        dataset[i * run_rows : (i+1) * run_rows, :] = data

    bar.finish()
    
    np.save(opts.path, dataset.astype(np.float32))

if __name__ == "__main__": main()
