import numpy as np
import numba
from numba.experimental import jitclass
from numba.typed import Dict

spec = [
    ('lr', numba.float32),
    ('_initialized', numba.boolean)
]


# @jitclass(spec=spec)
class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr

        self._initialized = False

    def update(self, param, grad_param, name):
        raise NotImplementedError

    def initialize_parameters(self, params):
        raise NotImplementedError





