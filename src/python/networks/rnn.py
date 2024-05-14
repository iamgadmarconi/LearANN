import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.input = x
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
            return z

    def backward(self, grad_output):
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)
        if self.activation == 'relu':
            grad_z = grad_output * (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = self.apply_activation(self.z)
            grad_z = grad_output * sig * (1 - sig)
        elif self.activation == 'tanh':
            tanh = self.apply_activation(self.z)
            grad_z = grad_output * (1 - tanh ** 2)
        else:
            grad_z = grad_output

        grad_z = grad_z.reshape(self.output_size, -1)
        self.grad_weights = np.dot(grad_z, self.input.T)
        self.grad_biases = grad_z.sum(axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_z)
        return grad_input

class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr

class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def initialize_moments(self, shape, name):
        if name not in self.m:
            self.m[name] = np.zeros(shape)
            self.v[name] = np.zeros(shape)

    def update(self, param, grad_param, name):
        self.initialize_moments(param.shape, name)
        self.t += 1
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad_param
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad_param ** 2)
        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param

class RNN:
    def __init__(self, layers_config):
        self.layers = [Layer(config['input_size'], config['output_size'], config['activation']) for config in layers_config]
        self.optimizer = Adam(lr=0.01)  # Increased learning rate

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
        for i, layer in enumerate(self.layers):
            layer.weights = self.optimizer.update(layer.weights, layer.grad_weights, f'layer_{i}_weights')
            layer.biases = self.optimizer.update(layer.biases, layer.grad_biases, f'layer_{i}_biases')

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = ((outputs - targets) ** 2).mean()
        grad_outputs = 2 * (outputs - targets) / outputs.size
        self.backward(grad_outputs)
        self.update_weights()
        return loss


def test_rnn():
    np.random.seed(42)  # For reproducibility

    layers_config = [
        {'input_size': 5, 'output_size': 10, 'activation': 'relu'},
        {'input_size': 10, 'output_size': 100, 'activation': 'relu'},
        {'input_size': 100, 'output_size': 1, 'activation': 'linear'}
    ]

    rnn = RNN(layers_config)
    
    # Example training data
    inputs = np.random.randn(5, 1)
    targets = np.random.randn(1, 1)
    
    # Training loop
    for epoch in range(1000):
        loss = rnn.train(inputs, targets)
        #if epoch % 100 == 0:
        #   print(f'Epoch {epoch}, Loss: {loss}')
        print(f'Epoch {epoch}, Loss: {loss}')

    # Test forward pass
    output = rnn.forward(inputs)
    print("Output after training:", output)

def generate_sine_wave_data(seq_length, num_sequences):
    x = np.linspace(0, 2 * np.pi, seq_length * num_sequences)
    y = np.sin(x)
    X = np.array([y[i:i+seq_length] for i in range(num_sequences)])
    y = np.array([y[i+1:i+seq_length+1] for i in range(num_sequences)])
    return X, y

def plot_sine_wave(true_data, predicted_data):
    plt.figure(figsize=(10, 6))
    plt.plot(true_data.flatten(), label="True Sine Wave")
    plt.plot(predicted_data.flatten(), label="Predicted Sine Wave")
    plt.legend()
    plt.show()

def test_rnn_on_sine_wave():
    # Generate sine wave data
    seq_length = 50
    num_sequences = 1000
    X, y = generate_sine_wave_data(seq_length, num_sequences)
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    # Configure and create RNN
    layers_config = [
        {'input_size': seq_length, 'output_size': 50, 'activation': 'relu'},
        {'input_size': 50, 'output_size': 50, 'activation': 'relu'},
        {'input_size': 50, 'output_size': 100, 'activation': 'tanh'},
        {'input_size': 100, 'output_size': 50, 'activation': 'tanh'},
        {'input_size': 50, 'output_size': seq_length, 'activation': 'linear'},
    ]
    rnn = RNN(layers_config)

    # Training loop
    for epoch in range(1000):
        total_loss = 0
        for i in range(len(X_train)):
            loss = rnn.train(X_train[i].reshape(-1, 1), y_train[i].reshape(-1, 1))
            total_loss += loss
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(X_train)}')

    # Testing loop
    predictions = []
    for i in range(len(X_test)):
        prediction = rnn.forward(X_test[i].reshape(-1, 1))
        predictions.append(prediction.flatten())

    predictions = np.array(predictions)

    # Plot the results
    plot_sine_wave(y_test, predictions)

# Run the test
if __name__ == "__main__":
    test_rnn_on_sine_wave()