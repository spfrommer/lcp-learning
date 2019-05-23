import numpy as np
from collections import namedtuple
import pdb

import lemkelcp

# x corresponds to sliding velocity, u external force
PhysicsParams = namedtuple('sliding_physics_params', 'us g mu dt')
SimParams = namedtuple('sliding_sim_params', 'x0 xdot0 time_steps')
# Outputs slack variables (lambdas are "slack" force differences from max)
SimSolution = namedtuple('sliding_solution', 'xs, posxdots, negxdots, poslambdas, neglambdas, us')
# xdots and lambdas are signed and lambdas represent actual friction forces
SimSolutionProcessed = namedtuple('sliding_solution_processed', 'xs, xdots, lambdas, us')
SimData = namedtuple('falling_data', 'xdots, us, lambdas, next_xdots')
# TODO: add full sim data

MARSHALLED_SIZE = 4
HAS_PROCESSING = True

def lcp(posxdot, negxdot, u, pp):
    M = (1/pp.dt) * np.eye(2)
    q = np.array([(1/pp.dt) * negxdot - (1/pp.dt) * posxdot - u + pp.mu * pp.g,
                  (1/pp.dt) * posxdot - (1/pp.dt) * negxdot + u + pp.mu * pp.g])
    return M, q

def slack_calc(xdotsol, M, q):
    # Solver doesn't output slack variable
    # So we calculate it explicitely
    pos_zero = np.isclose(xdotsol[0], 0) 
    # Pos velocity and negative slack friction are complementary
    neglambda = np.dot(M[0,:], xdotsol) + q[0] if pos_zero else 0
    
    neg_zero = np.isclose(xdotsol[1], 0)
    # Negative velocity and positive slack friction are complementary
    poslambda = np.dot(M[1,:], xdotsol) + q[1] if neg_zero else 0

    return poslambda, neglambda

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
    # Lambda corresponds roughly to unused friction force, see notes
    # Just initialize to zero, depends on previous dynamics vars
    poslambdas = np.zeros((sp.time_steps, 1))
    neglambdas = np.zeros((sp.time_steps, 1))

    for t in range(sp.time_steps-1):
        M, q = lcp(posxdots[t], negxdots[t], pp.us[t+1], pp)
        xdotsol, exit_code, _ = lemkelcp.lemkelcp(M, q)
        assert(exit_code == 0)
        
        posxdots[t+1] = xdotsol[0]
        negxdots[t+1] = xdotsol[1]
        xs[t+1] = dynamics_step(xs[t], posxdots[t+1], negxdots[t+1], pp)
        poslambdas[t+1], neglambdas[t+1] = slack_calc(xdotsol, M, q)
    
    # Make sure both velocities can't be positive at the same time
    assert((np.isclose(posxdots * negxdots, 0)).all())

    return SimSolution(xs, posxdots, negxdots, poslambdas, neglambdas, pp.us)

def process_solution(ss, pp):
    xdots = ss.posxdots - ss.negxdots;
    # Pos/neg friction forces (solution lambdas are actually slack forces)
    lambda_forces = pp.mu * pp.g - np.hstack((ss.poslambdas, ss.neglambdas))
    # One force will usually be positive, one negative
    # We want the positive one and the associated sign
    lambdas = np.expand_dims(np.amax(lambda_forces, axis=1), axis=1)
    lambda_signs = np.expand_dims(lambda_forces.argmax(axis=1), axis=1)
    lambda_signs[lambda_signs == 1] = -1
    lambda_signs[lambda_signs == 0] = 1
    lambdas = lambdas * lambda_signs
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
    pp = PhysicsParams(us = (1) * np.ones(30), g=1.0, mu=3.0, dt=1.0)
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
