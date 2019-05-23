import numpy as np
from collections import namedtuple
import pdb

import lemkelcp

from lcplearn import lcp_utils

# Traditional formulation of sliding block LCP, assuming m=1, dt=1

# x corresponds to sliding velocity, u external force
PhysicsParams = namedtuple('sliding_physics_params', 'us g mu')
SimParams = namedtuple('sliding_sim_params', 'x0 xdot0 time_steps')
SimSolution = namedtuple('sliding_solution', 'xs, xdots, poslambdas, neglambdas, gammas, us')
# xdots and lambdas are signed and lambdas represent actual friction forces
SimSolutionProcessed = namedtuple('sliding_solution_processed', 'xs, xdots, lambdas, gammas, us')
SimData = namedtuple('falling_data', 'xdots, us, lambdas, gammas, next_xdots')
# TODO: add full sim data

MARSHALLED_SIZE = 5
HAS_PROCESSING = True

def lcp(xdot, u, pp):
    M = np.array([[1, -1, 1], [-1, 1, 1], [-1, -1, 0]])
    q = np.array([xdot + u, -xdot -u, pp.g * pp.mu])
    return M, q

def dynamics_step(x, xdot, poslambda, neglambda, u):
    xdot = xdot + poslambda - neglambda + u
    x = x + xdot
    return x, xdot

def sim(pp, sp):
    xs = np.zeros((sp.time_steps, 1))
    xs[0] = sp.x0
    # TODO: formulate these as vectors
    xdots = np.zeros((sp.time_steps, 1))
    xdots[0] = sp.xdot0
    poslambdas = np.zeros((sp.time_steps, 1))
    neglambdas = np.zeros((sp.time_steps, 1))
    gammas = np.zeros((sp.time_steps, 1))

    for t in range(sp.time_steps-1):
        M, q = lcp(xdots[t], pp.us[t+1], pp)
        sol, exit_code, _ = lemkelcp.lemkelcp(M, q)
        assert(exit_code == 0)
        
        poslambdas[t+1] = sol[0]
        neglambdas[t+1] = sol[1]
        gammas[t+1] = sol[2]

        xs[t+1], xdots[t+1] = dynamics_step(xs[t], xdots[t],
                poslambdas[t+1], neglambdas[t+1], pp.us[t+1])
    
    return SimSolution(xs, xdots, poslambdas, neglambdas, \
                       gammas, np.expand_dims(pp.us, 1))

def process_solution(ss, pp):
    lambdas = ss.poslambdas - ss.neglambdas
    return SimSolutionProcessed(ss.xs, ss.xdots, lambdas, ss.gammas, ss.us)

# Lines up data for machine learning
def marshal_data(ss):
    return np.hstack((ss.xdots[:-1], ss.us[1:],
                      ss.lambdas[1:], ss.xdots[1:]))

# numpy array -> sd
def unmarshal_data(data):
    return SimData(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

def main():
    print('Solving sliding sim...')
    pp = PhysicsParams(us = (-4) * np.ones(30), g=1.0, mu=3.0)
    sp = SimParams(x0=0.0, xdot0=0, time_steps=30)
    sol = sim(pp, sp)
    print('Full output (ignore first lambda/gamma): ')
    print('x, xdot, lambda+, lambda-, gamma, u')
    print(np.hstack((sol.xs, sol.xdots, 
                     sol.poslambdas, sol.neglambdas,
                     sol.gammas, sol.us)))
    sol_processed = process_solution(sol, pp)
    print('Processed output (ignore first lambda): ')
    print('x, xdot, lambda, gamma, u')
    print(np.hstack((sol_processed.xs, sol_processed.xdots,
                     sol_processed.lambdas, sol_processed.gammas,
                     sol_processed.us)))
