from utils.optimizers import Adam, Adagrad, GradientDescent
from utils.layers import Layer

class RNN:
    def __init__(self, layers_config, optimizer_name: str = 'adam', optimizer_params: dict = {'lr': 0.01}):
        self.optimizer = self._create_optimizer(optimizer_name, optimizer_params)
        self.layers = [Layer(config['input_size'], config['output_size'], config['activation']) for config in layers_config]

    def _create_optimizer(self, optimizer_name, optimizer_params):
        name = optimizer_name.lower()

        if name == 'gradientdescent':
            return GradientDescent(**optimizer_params)
        elif name == 'adagrad':
            return Adagrad(**optimizer_params)
        elif name == 'adam':
            return Adam(**optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

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
