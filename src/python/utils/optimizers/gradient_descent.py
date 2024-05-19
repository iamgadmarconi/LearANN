import numpy as np
import numba

from utils.optimizers.optimizers import Optimizer

class GradientDescent(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)

    @numba.njit
    def update(self, param, grad_param, name):
        param -= self.learning_rate * grad_param
        return param
