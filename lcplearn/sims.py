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
    SLIDING_TRADITIONAL_LCP = 'sliding_traditional_lcp'
    SLIDING_TRADITIONAL_MASS = 'sliding_traditional_mass'
    SLIDING_TRADITIONAL_BASIS = 'sliding_traditional_basis'

    def __str__(self):
        return self.value

class AnalyzeType(Enum):
    FALLING = 'falling'
    SLIDING_DIRECT = 'sliding_direct'
    SLIDING_TRADITIONAL_DATA = 'sliding_traditional_data'
    SLIDING_TRADITIONAL_LCP = 'sliding_traditional_lcp'
    SLIDING_TRADITIONAL_MASS = 'sliding_traditional_mass'
    SLIDING_TRADITIONAL_BASIS = 'sliding_traditional_basis'

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
    import sliding.traditional.lcp_structured_model as SlidingTraditionalLcpModel
    import sliding.traditional.mass_estimate_model as SlidingTraditionalMassModel
    import sliding.traditional.basis_model as SlidingTraditionalBasisModel

    if modeltype == ModelType.FALLING:
        return FallingModel
    elif modeltype == ModelType.SLIDING_DIRECT:
        return SlidingDirectModel
    elif modeltype == ModelType.SLIDING_TRADITIONAL_LCP:
        return SlidingTraditionalLcpModel
    elif modeltype == ModelType.SLIDING_TRADITIONAL_MASS:
        return SlidingTraditionalMassModel
    elif modeltype == ModelType.SLIDING_TRADITIONAL_BASIS:
        return SlidingTraditionalBasisModel

def analyze_module(modeltype):
    import falling.analyze as FallingAnalyze
    import sliding.direct.analyze as SlidingDirectAnalyze
    import sliding.traditional.data_analyze as SlidingTraditionalDataAnalyze
    import sliding.traditional.lcp_analyze as SlidingTraditionalLcpAnalyze
    import sliding.traditional.basis_analyze as SlidingTraditionalBasisAnalyze

    if modeltype == AnalyzeType.FALLING:
        return FallingAnalyze
    elif modeltype == AnalyzeType.SLIDING_DIRECT:
        return SlidingDirectAnalyze
    elif modeltype == AnalyzeType.SLIDING_TRADITIONAL_DATA:
        return SlidingTraditionalDataAnalyze
    elif modeltype == AnalyzeType.SLIDING_TRADITIONAL_LCP:
        return SlidingTraditionalLcpAnalyze
    elif modeltype == AnalyzeType.SLIDING_TRADITIONAL_MASS:
        return SlidingTraditionalLcpAnalyze
    elif modeltype == AnalyzeType.SLIDING_TRADITIONAL_BASIS:
        return SlidingTraditionalBasisAnalyze
    else:
        print("Can't find model for: " + str(modeltype))
