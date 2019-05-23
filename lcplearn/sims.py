import numpy as np
from collections import namedtuple
from enum import Enum

class SimType(Enum):
    FALLING = 'falling'
    SLIDING_DIRECT = 'sliding_direct'
    SLIDING_TRADITIONAL = 'sliding_traditional'

    def __str__(self):
        return self.value

class ModelType(Enum):
    FALLING = 'falling'
    SLIDING_DIRECT = 'sliding_direct'

    def __str__(self):
        return self.value

def dynamics_module(simtype):
    import falling.dynamics as FallingDynamics
    import sliding.direct.dynamics as SlidingDirectDynamics
    import sliding.traditional.dynamics as SlidingTraditionalDynamics

    if simtype == SimType.FALLING:
        return FallingDynamics
    elif simtype == SimType.SLIDING_DIRECT:
        return SlidingDirectDynamics
    elif simtype == SimType.SLIDING_TRADITIONAL:
        return SlidingTraditionalDynamics

def model_module(modeltype):
    import falling.model as FallingModel
    import sliding.direct.model as SlidingDirectModel

    if modeltype == ModelType.FALLING:
        return FallingModel
    elif modeltype == ModelType.SLIDING_DIRECT:
        return SlidingDirectModel

def analyze_module(modeltype):
    import falling.analyze as FallingAnalyze
    import sliding.direct.analyze as SlidingDirectAnalyze

    if modeltype == ModelType.FALLING:
        return FallingAnalyze
    elif modeltype == ModelType.SLIDING_DIRECT:
        return SlidingDirectAnalyze


