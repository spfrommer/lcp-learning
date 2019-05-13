import sys
import math
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from enum import Enum
import pdb

import lemkelcp as lcp

# x corresponds to height
FallingSimParams = namedtuple('falling_params', 'g x0 xdot0 lambda0 dt time_steps')
FallingSimSolution = namedtuple('falling_solution', 'xs, xdots, lambdas')
# x corresponds to sliding velocity, u external force
SlidingSimParams = namedtuple('sliding_params', 'g x0 xdot0 mu dt us time_steps')
# Outputs slack variables (lambdas are "slack" force differences from max)
SlidingSimSolution = namedtuple('sliding_solution', 'xs, posxdots, negxdots, poslambdas, neglambdas')
# xdots and lambdas are signed and lambdas represent actual friction forces
SlidingSimSolutionProcessed = namedtuple('sliding_solution_processed', 'xs, xdots, lambdas')

class SimType(Enum):
    FALLING = 'falling'
    SLIDING = 'sliding'

    def __str__(self):
        return self.value

def main():
    parser = ArgumentParser()
    parser.add_argument('simtype', type=SimType, choices=list(SimType))
    opts = parser.parse_args()

    if opts.simtype == SimType.FALLING:
        print('Solving falling sim...')
        p = FallingSimParams(x0=20.0,  xdot0=4.0,  lambda0=0.0,
                      g=1.0,    dt=1.0,     time_steps=30)
        sol = falling_box_sim(p)
        print('x, xdot, lambda')
        print(np.hstack((sol.xs, sol.xdots, sol.lambdas)))

    if opts.simtype == SimType.SLIDING:
        print('Solving sliding sim...')
        p = SlidingSimParams(x0=0.0,  xdot0=0,  mu=3.0,
                      g=1.0,   dt=1.0,     time_steps=30,
                      us = (1) * np.ones(30))
        sol = sliding_box_sim(p)
        print('Full output (ignore first lambda): ')
        print('x, xdot+, xdot-, lambda\'+, lambda\'-')
        print(np.hstack((sol.xs, sol.posxdots, sol.negxdots,
                         sol.poslambdas, sol.neglambdas)))
        sol_processed = process_sliding_solution(sol, p)
        print('Processed output (ignore first lambda): ')
        print('x, xdot, lambda')
        print(np.hstack((sol_processed.xs, sol_processed.xdots,
                         sol_processed.lambdas)))

# Takes in parameters as a FallingSimParams
def falling_box_sim(p):
    xs = np.zeros((p.time_steps, 1))
    xs[0] = p.x0
    xdots = np.zeros((p.time_steps, 1))
    xdots[0] = p.xdot0
    lambdas = np.zeros((p.time_steps, 1))
    lambdas[0] = p.lambda0

    for t in range(p.time_steps-1):
        M = np.array([[(1/p.dt)**2]])
        q = np.array([-(1/(p.dt**2)) *
            (xs[t] + xdots[t] * p.dt - p.g * (p.dt**2))])
        xsol, exit_code, _ = lcp.lemkelcp(M, q)
        assert(exit_code == 0)
        
        xs[t+1] = xsol

        # Solver doesn't output slack variable
        # So we calculate it explicitely (otherwise just zero)
        if np.isclose(xsol, 0):
            lambdas[t+1] = M[0,0] * xsol + q[0]
            
        xdots[t+1] = xdots[t] + (-p.g + lambdas[t+1]) * p.dt
    
    return FallingSimSolution(xs, xdots, lambdas)

# Takes in parameters as a SlidingSimParams
def sliding_box_sim(p):
    xs = np.zeros((p.time_steps, 1))
    xs[0] = p.x0
    posxdots = np.zeros((p.time_steps, 1))
    posxdots[0] = p.xdot0 * (np.sign(p.xdot0) > 0)
    negxdots = np.zeros((p.time_steps, 1))
    negxdots[0] = (-p.xdot0) * (np.sign(p.xdot0) < 0)
    # Lambda corresponds roughly to unused friction force, see notes
    # Just initialize to zero, depends on previous dynamics vars
    poslambdas = np.zeros((p.time_steps, 1))
    neglambdas = np.zeros((p.time_steps, 1))

    for t in range(p.time_steps-1):
        M = (1/p.dt) * np.eye(2)
        q = np.array([(1/p.dt) * negxdots[t] - (1/p.dt) * posxdots[t] - p.us[t+1] + p.mu * p.g,
                      (1/p.dt) * posxdots[t] - (1/p.dt) * negxdots[t] + p.us[t+1] + p.mu * p.g])
        xdotsol, exit_code, _ = lcp.lemkelcp(M, q)
        assert(exit_code == 0)
        
        posxdots[t+1] = xdotsol[0]
        negxdots[t+1] = xdotsol[1]
        xs[t+1] = xs[t] + p.dt * (posxdots[t+1] - negxdots[t+1])
        
        # Solver doesn't output slack variable
        # So we calculate it explicitely
        if np.isclose(xdotsol[0], 0):
            # Pos velocity and negative friction are complementary
            neglambdas[t+1] = np.dot(M[0,:], xdotsol) + q[0]
        if np.isclose(xdotsol[1], 0):
            # Negative velocity and positive friction are complementary
            poslambdas[t+1] = np.dot(M[1,:], xdotsol) + q[1]
    
    # Make sure both velocities can't be positive at the same time
    assert((np.isclose(posxdots * negxdots, 0)).all())

    return SlidingSimSolution(xs, posxdots, negxdots, poslambdas, neglambdas)

def process_sliding_solution(sol, params):
    xdots = sol.posxdots - sol.negxdots;
    # Pos/neg friction forces (solution lambdas are actually slack forces)
    lambda_forces = params.mu * params.g - np.hstack((sol.poslambdas, sol.neglambdas))
    # One force will usually be positive, one negative
    # We want the positive one and the associated sign
    lambdas = np.expand_dims(np.amax(lambda_forces, axis=1), axis=1)
    lambda_signs = np.expand_dims(lambda_forces.argmax(axis=1), axis=1)
    lambda_signs[lambda_signs == 1] = -1
    lambda_signs[lambda_signs == 0] = 1
    lambdas = lambdas * lambda_signs
    return SlidingSimSolutionProcessed(sol.xs, xdots, lambdas)

if __name__ == "__main__": main()
