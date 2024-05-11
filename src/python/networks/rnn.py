import numpy as np
import matplotlib.pyplot as plt


class RNN:
    def __init__(self, layers_config):
        self.layers = [Layer(config['input_size'], config['output_size'], config['activation']) for config in layers_config]
        self.optimizer = Adam(lr=0.001)

    def forward(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, grad_output):
        grad_input = grad_output
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer.weights, layer.grad_weights)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = ((outputs - targets) ** 2).mean()
        grad_outputs = 2 * (outputs - targets) / outputs.size
        self.backward(grad_outputs)
        self.update_weights()
        return loss
    

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros(output_size)
        self.activation = activation

    def forward(self, x):
        self.input = x  # Store the input for use in the backward pass
        self.z = np.dot(self.weights, x) + self.biases
        return self.apply_activation(self.z)

    def apply_activation(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            return z  # Linear activation as default

    def backward(self, grad_output):
        # Calculate gradient of activation
        if self.activation == 'relu':
            grad_z = grad_output * (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = self.apply_activation(self.z)
            grad_z = grad_output * sig * (1 - sig)
        elif self.activation == 'tanh':
            tanh = self.apply_activation(self.z)
            grad_z = grad_output * (1 - tanh ** 2)
        else:
            grad_z = grad_output  # Linear activation

        # Ensure grad_z is 2D
        grad_z = grad_z.reshape(self.output_size, -1)  # Correct reshaping to ensure 2D array
        
        # Compute gradients with respect to weights and biases correctly matching the dimensions of weights
        self.grad_weights = np.dot(grad_z, self.input.T)  # Ensure this matches weight dimensions
        self.grad_biases = grad_z.sum(axis=1)  # Sum across batch dimension for biases
        # Compute gradient with respect to input to this layer
        grad_input = np.dot(self.weights.T, grad_z)
        return grad_input.squeeze() 


class Optimizer:

    def __init__(self, lr) -> None:
        self.learing_rate = lr

    def update(self):
        raise NotImplementedError


class Adam(Optimizer):

    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        super().__init__(lr)

        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon

        self.m = None  
        self.v = None
        self.t = 0

    def update(self, weights, grad_weights):
        if self.m is None:  # Initialize moments m and v if they are None
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        # Increment time step
        self.t += 1
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_weights
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_weights ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update weights
        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def test_rnn_corrected():
    # Define a simple sequential input and a shifted output, reshaping to (10, 1, 1) for individual timestep processing
    input_sequence = np.array([[[i]] for i in range(1, 11)])  # Shape (10, 1, 1)
    target_sequence = np.array([[[i]] for i in range(2, 12)])  # Shape (10, 1, 1)

    # RNN configuration
    layers_config = [
        {'input_size': 1, 'output_size': 5, 'activation': 'relu'},
        {'input_size': 5, 'output_size': 1, 'activation': 'linear'}  # Output layer with linear activation
    ]
    
    # Initialize RNN
    rnn = RNN(layers_config)

    # Training loop
    epochs = 500  # Reduce epochs for quicker testing
    for epoch in range(epochs):
        loss = 0
        for t in range(input_sequence.shape[0]):  # Process each timestep individually
            output = rnn.forward(input_sequence[t])
            loss += ((output - target_sequence[t]) ** 2).mean()
            grad_outputs = 2 * (output - target_sequence[t]) / output.size
            rnn.backward(grad_outputs)
            rnn.update_weights()
        loss /= input_sequence.shape[0]
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss}')

    # Test the trained RNN on the full sequence
    outputs = np.array([rnn.forward(input_sequence[t]) for t in range(input_sequence.shape[0])])
    print("Expected Output:", target_sequence.flatten())
    print("RNN Output:", outputs.flatten())

# Uncomment below line to run the corrected test
# test_rnn_corrected()


test_rnn_corrected()