import numpy as np
from collections import namedtuple
import pdb

import lemkelcp

# x corresponds to sliding velocity, u external force
PhysicsParams = namedtuple('sliding_physics_params', 'us g mu dt')
SimParams = namedtuple('sliding_sim_params', 'x0 xdot0 time_steps')
# Outputs slack variables (lambdas are actual friction forces)
SimSolution = namedtuple('sliding_solution', 'xs, posxdots, negxdots, poslambdas, neglambdas, us')
# xdots and lambdas are signed
SimSolutionProcessed = namedtuple('sliding_solution_processed', 'xs, xdots, lambdas, us')
SimData = namedtuple('falling_data', 'xdots, us, lambdas, next_xdots')

MARSHALLED_SIZE = 4
HAS_PROCESSING = True

def lcp(posxdot, negxdot, u, pp):
    M = np.array([[0,        -pp.dt, 0],
                  [-pp.dt,   0,      0],
                  [-1,       -1,     0]])
    q = np.array([posxdot - negxdot + max(pp.dt * u, 0),
                  negxdot - posxdot - min(pp.dt * u, 0),
                  pp.mu * pp.g])
    return M, q

def slack_calc(lambdasol, M, q):
    # Solver doesn't output slack variable
    # So we calculate it explicitely
    pos_zero = np.isclose(lambdasol[0], 0) 
    # Pos velocity and pos friction are complementary
    posxdot = np.dot(M[0,:], lambdasol) + q[0] if pos_zero else 0
    
    neg_zero = np.isclose(lambdasol[1], 0)
    # Negative velocity and negative friction are complementary
    negxdot = np.dot(M[1,:], lambdasol) + q[1] if neg_zero else 0

    return posxdot, negxdot

def dynamics_step(x, posxdot, negxdot, pp):
    return x + pp.dt * (posxdot - negxdot)

def sim(pp, sp):
    xs = np.zeros((sp.time_steps, 1))
    xs[0] = sp.x0
    # TODO: formulate these as vectors
    posxdots = np.zeros((sp.time_steps, 1))
    posxdots[0] = sp.xdot0 * (np.sign(sp.xdot0) > 0)
    negxdots = np.zeros((sp.time_steps, 1))
    negxdots[0] = (-sp.xdot0) * (np.sign(sp.xdot0) < 0)
    poslambdas = np.zeros((sp.time_steps, 1))
    neglambdas = np.zeros((sp.time_steps, 1))

    for t in range(sp.time_steps-1):
        M, q = lcp(posxdots[t], negxdots[t], pp.us[t+1], pp)
        lambdasol, exit_code, _ = lemkelcp.lemkelcp(M, q)
        pdb.set_trace()
        assert(exit_code == 0)
        
        poslambdas[t+1] = lambdasol[0]
        neglambdas[t+1] = lambdasol[1]

        posxdots[t+1], negxdots[t+1] = slack_calc(lambdasol, M, q)
        xs[t+1] = dynamics_step(xs[t], posxdots[t+1], negxdots[t+1], pp)
    
    # Make sure both velocities can't be positive at the same time
    assert((np.isclose(posxdots * negxdots, 0)).all())

    return SimSolution(xs, posxdots, negxdots, poslambdas, neglambdas, pp.us)

def process_solution(ss, pp):
    xdots = ss.posxdots - ss.negxdots;
    lambdas = ss.poslambdas - ss.neglambdas;
    return SimSolutionProcessed(ss.xs, xdots, lambdas, ss.us)

# Lines up data for machine learning
def marshal_data(ss):
    return np.hstack((ss.xdots[:-1], ss.us[1:],
                      ss.lambdas[1:], ss.xdots[1:]))

# numpy array -> sd
def unmarshal_data(data):
    return SimData(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

def main():
    print('Solving sliding sim...')
    pp = PhysicsParams(us = (30) * np.ones(30), g=1.0, mu=3.0, dt=1.0)
    #pp = PhysicsParams(us = np.arange(-3, 27), g=1.0, mu=3.0, dt=1.0)
    sp = SimParams(x0=0.0, xdot0=0, time_steps=30)
    sol = sim(pp, sp)
    print('Full output (ignore first lambda): ')
    print('x, xdot+, xdot-, lambda\'+, lambda\'-')
    print(np.hstack((sol.xs, sol.posxdots, sol.negxdots,
                     sol.poslambdas, sol.neglambdas)))
    sol_processed = process_solution(sol, pp)
    print('Processed output (ignore first lambda): ')
    print('x, xdot, lambda')
    print(np.hstack((sol_processed.xs, sol_processed.xdots,
                     sol_processed.lambdas)))
