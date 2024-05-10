import numpy as np

class Kernel:

    def __init__(self, size, dim) -> None:
        self._dim = dim
        self._size = size

    @property
    def size(self):
        return self._size
    
    @property
    def dim(self):
        return self._dim

    def __call__(self, x):
        return self._kernel(x)
    
    def _kernel(self, x):
        raise NotImplementedError
    

class GaussianKernel(Kernel):
    def __init__(self, size, dim, sigma) -> None:
        super().__init__(size, dim)
        self._sigma = sigma
        self._kernel = self._gaussian_kernel

    def _gaussian_kernel(self, x):
        """
        Compute the Gaussian kernel for the given data

        Args:
        x: np.ndarray
            The input data

        Returns:
        np.ndarray
            The kernel values
        """
        return np.exp(-np.sum(x**2, axis=1) / (2 * self._sigma**2))