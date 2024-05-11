import numpy as np


input_size = 10
hidden_size = 5
output_size = 1

layers_config = [
    {'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size, 'activation': 'relu'},
    {'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size, 'activation': 'tanh'},
    {'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size, 'activation': 'sigmoid'},
]


class RNN:

    def __init__(self, layers_config, optimizer='adam', learning_rate=0.001) -> None:
        self.layers = [Layer(config['input_size'], config['hidden_size'], config['output_size'], config['activation']) for config in layers_config]

        if optimizer.lower() == 'adam':
            self.optimizer = Adam(lr=learning_rate)

    def forward(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, grad):
        grad_input = grad
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)

    def update(self):
        """Update the weights of the network using the gradients computed in the backward pass.
        
        Args:
            lr (float): The learning rate to use for the update.
        """
        for layer in self.layers:
            self.optimizer.update(layer.weights, layer.grad_weights)

    def train(self, inputs, targets):
        """Train the network on a single batch of data.
        
        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The target data.
            lr (float): The learning rate to use for the update.
        """
        outputs = self.forward(inputs)
        loss = ((outputs - targets) ** 2).mean()
        grad_outputs = 2 * (outputs - targets) / outputs.size
        self.backward(grad_outputs)
        self.update()
        return loss
    

class Layer:

    def __init__(self, input_size, output_size, activation: str='relu') -> None:
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)

        self.biases = np.zeros(output_size)
        self.activation = activation

    def forward(self, x):
        self.z = np.dot(self.weights, x) + self.biases
        return self.apply_activation(self.z)

    def apply_activation(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        return z

    def backward(self, grad):
        if self.activation == 'relu':
            grad_z = grad * (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = self.apply_activation(self.z)
            grad_z = grad * sig * (1 - sig)
        elif self.activation == 'tanh':
            tanh = self.apply_activation(self.z)
            grad_z = grad * (1 - tanh ** 2)
        else:
            grad_z = grad

        self.grad_weights = np.outer(grad_z, )

    def update(self, lr):
        pass

    def train(self, x, y, lr):
        pass


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
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_weights
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_weights**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        weights -= self.learing_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
