from utils.optimizers import Adam, Adagrad, GradientDescent
from utils.layers import Layer, GPULayer

class RNN:

    def __init__(self, layers_config, optimizer_name='adam', optimizer_params=None, cuda=False):
        if cuda:
            self.layers = [GPULayer(config['input_size'], config['output_size'], config['activation']) for config in layers_config]
        else:
            self.layers = [Layer(config['input_size'], config['output_size'], config['activation']) for config in layers_config]

        if optimizer_params is None:
            optimizer_params = {'lr': 0.01}

        self.optimizer = self._create_optimizer(optimizer_name, optimizer_params)

    def _create_optimizer(self, optimizer_name, optimizer_params):

        name = optimizer_name.lower()

        print(f"Creating optimizer: {name}")

        if name == 'gradientdescent':
            optimizer = GradientDescent(**optimizer_params)
            
        elif name == 'adagrad':
            optimizer = Adagrad(**optimizer_params)
            param_shapes = {f'layer_{i}_weights': layer.weights.shape for i, layer in enumerate(self.layers)}
            param_shapes.update({f'layer_{i}_biases': layer.biases.shape for i, layer in enumerate(self.layers)})
            optimizer.initialize_parameters(param_shapes)

        elif name == 'adam':
            optimizer = Adam(**optimizer_params)
            param_shapes = {f'layer_{i}_weights': layer.weights.shape for i, layer in enumerate(self.layers)}
            param_shapes.update({f'layer_{i}_biases': layer.biases.shape for i, layer in enumerate(self.layers)})
            optimizer.initialize_parameters(param_shapes)

        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

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
