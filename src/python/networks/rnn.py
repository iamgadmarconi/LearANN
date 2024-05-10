import random
import numpy as np
import numba

spec = [
    ('_input_size', numba.int32),
    ('_hidden_size', numba.int32),
    ('_output_size', numba.int32),
    ('_Wxh', numba.float64[:, :]),  # Weight matrix for input to hidden
    ('_Whh', numba.float64[:, :]),  # Weight matrix for hidden to hidden
    ('_Why', numba.float64[:, :]),  # Weight matrix for hidden to output
    ('_bh', numba.float64[:]),      # Hidden bias
    ('_by', numba.float64[:]),      # Output bias
]

@numba.experimental.jitclass(spec)
class RNN:

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the RNN model

        Args:
        input_size: int
            The size of the input layer
        hidden_size: int
            The size of the hidden layer
        output_size: int
            The size of the output layer

        Returns:
        None


        """
        # The size of the input, hidden, and output layers
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        # The RNN has input to hidden connections parameterized by a weight matrix Wxh,
        # hidden-to-hidden recurrent connections parameterized by a weight matrix Whh, 
        # and hidden-to-output connections parameterized by a weight matrix Why
        # All these weights (Wxh, Whh, Why) are shared across time.

        # Wxh: Weight matrix for input to hidden
        self._Wxh = np.random.randn(hidden_size, input_size)

        # Whh: Weight matrix for hidden to hidden
        self._Whh = np.random.randn(hidden_size, hidden_size)

        # Why: Weight matrix for hidden to output
        self._Why = np.random.randn(output_size, hidden_size)

        # The bias is an "offset" added to each unit in a neural network layer that's independent of the input to the layer. 
        # The bias permits a layer to model a data space that's centered around some point other than the origin.

        # bh: Hidden bias
        self._bh = np.zeros(hidden_size)

        # by: Output bias
        self._by = np.zeros(output_size)

    @property
    def input_size(self):
        """
        The size of the input layer
        """
        return self._input_size
    
    @property
    def hidden_size(self):
        """
        The size of the hidden layer
        """
        return self._hidden_size
    
    @property
    def output_size(self):
        """
        The size of the output layer
        """
        return self._output_size
    
    @property
    def Wxh(self):
        """
        The weight matrix for input to hidden
        """
        return self._Wxh
    
    @property
    def Whh(self):
        """
        The weight matrix for hidden to hidden
        """
        return self._Whh
    
    @property
    def Why(self):
        """
        The weight matrix for hidden to output
        """
        return self._Why    

    @property
    def bh(self):
        """
        The hidden bias
        """
        return self._bh 
    
    @property
    def by(self):
        """
        The output bias
        """
        return self._by

from numba import njit
import numpy as np

@njit
def rnn_step(Wxh, Whh, Why, bh, by, h, x):
    """
    Perform a single step of the RNN, ensuring all data are floating point for numba compatibility.
    """
    # Convert inputs to float if not already, and reshape to 2D column vectors
    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1, 1)
    h = np.ascontiguousarray(h, dtype=np.float64).reshape(-1, 1)

    # Compute the next hidden state and the output using float arrays
    h_next = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh.reshape(-1, 1))
    y = np.dot(Why, h_next) + by.reshape(-1, 1)

    return y.ravel(), h_next.ravel()

@njit
def forward(rnn, inputs):
    """
    Process a sequence of inputs through the RNN using numba's nopython mode.
    """
    # Initialize hidden state as a floating point 2D column vector
    h = np.zeros((rnn.hidden_size, 1), dtype=np.float64)
    outputs = []
    for x in inputs:
        # Ensure each input x is a contiguous floating point 2D column vector
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1, 1)
        y, h = rnn_step(rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by, h, x)
        outputs.append(y)
    return outputs, h.ravel()

# Define a simple test to run the forward function
def test_rnn():
    input_size = 10
    hidden_size = 20
    output_size = 5
    rnn = RNN(input_size, hidden_size, output_size)
    
    # Generate a simple sequence of 5 random float vectors
    inputs = np.random.randn(5, input_size).astype(np.float64)
    
    # Call forward to process the sequence
    outputs, final_hidden_state = forward(rnn, inputs)
    print("Outputs:", outputs)
    print("Final Hidden State:", final_hidden_state)

# Assuming RNN class definition is available and correctly defined
test_rnn()