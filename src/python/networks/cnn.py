import random
import numpy as np
import numba


spec = [
    ('value', numba.int32),               # a simple scalar field
    ('array', numba.float32[:]),          # an array field
]


@numba.experimental.jitclass(spec)
class Conv2D:

    def __init__(self) -> None:
        self.weights = np.random.randn(3, 3)
        self.bias = random.random()

    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def outputs(self, input):
        pass

