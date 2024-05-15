import numpy as np
from networks.rnn.rnn import RNN

class LSTM(RNN):

    def __call__(self, *args: np.Any, **kwds: np.Any) -> np.Any:
        return super().__call__(*args, **kwds)
    