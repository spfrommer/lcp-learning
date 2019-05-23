import numpy as np
from collections import namedtuple
from enum import Enum

# x corresponds to sliding velocity, u external force
SlidingSimParams = namedtuple('sliding_params', 'g x0 xdot0 mu dt us time_steps')
# Outputs slack variables (lambdas are "slack" force differences from max)
SlidingSimSolution = namedtuple('sliding_solution', 'xs, posxdots, negxdots, poslambdas, neglambdas, us')
# xdots and lambdas are signed and lambdas represent actual friction forces
SlidingSimSolutionProcessed = namedtuple('sliding_solution_processed', 'xs, xdots, lambdas, us')
SlidingSimData = namedtuple('falling_data', 'xdots, us, lambdas, next_xdots')

class SimType(Enum):
    FALLING = 'falling'
    SLIDING = 'sliding'

    def __str__(self):
        return self.value


# Similar idea but for sliding data, in this case we want to predict
# xdot in the next time step
def marshal_sliding_data(sol):
    return np.hstack((sol.xdots[:-1], sol.us[1:],
                      sol.lambdas[1:], sol.xdots[1:]))

def save_marshalled_data(data, path):
    np.save(path, data.astype(np.float32))

def unmarshal_sliding_data(data):
    return SlidingSimData(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

def load_sliding_data(path):
    return unmarshal_sliding_data(np.load(path))

def variable_count(simtype):
    if simtype == SimType.FALLING:
        return 3
    elif simtype == SimType.SLIDING:
        return 4
