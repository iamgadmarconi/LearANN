import numpy as np
from numba import jit

from utils.optimizers.optimizers import Optimizer

class GradientDescent(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)

    @jit(nopython=True, fastmath=True)
    def update(self, param, grad_param, name):
        param -= self.learning_rate * grad_param
        return param
