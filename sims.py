import numpy as np
from collections import namedtuple
from enum import Enum
import pdb

import falling.dynamics as FallingDynamics
import sliding.direct.dynamics as SlidingDirectDynamics

class SimType(Enum):
    FALLING = 'falling'
    SLIDING_DIRECT = 'sliding_direct'

    def __str__(self):
        return self.value

def dynamics_module(simtype):
    if simtype == SimType.FALLING:
        return FallingDynamics
    elif simtype == SimType.SLIDING_DIRECT:
        return SlidingDirectDynamics

def save_marshalled_data(data, path):
    np.save(path, data.astype(np.float32))

def load_marshalled_data(path, simtype):
    return dynamics_module(simtype).unmarshal_data(np.load(path))

def marshalled_size(simtype):
    return dynamics_module(simtype).MARSHALLED_SIZE
