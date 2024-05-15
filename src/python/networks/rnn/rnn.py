import numpy as np

from utils.optimizers import Adam, Adagrad, GradientDescent
from utils.layers import Layer, GPULayer, LSTMCell
from utils.cuda.cuda import gpu_mse_loss

class RNN:
    def __init__(self, layers_config, optimizer_name='adam', optimizer_params=None, **kwargs):
        """
        Constructor for the RNN class.

        :param layers_config: List of dictionaries containing the configuration for each layer.
        :param optimizer_name: Name of the optimizer to use one of ['gradientdescent', 'adagrad', 'adam'].
        :param optimizer_params: Dictionary containing the parameters for the optimizer. See the optimizer classes for more details.
        :param kwargs: Additional arguments.
            :param cuda: Boolean flag to indicate whether to use GPU or not.
            :param layer_type: Type of layer to use one of ['dense', 'lstm'], defaults to dense.
        """

        layer_type = kwargs.get('layer_type', 'dense').lower()
        use_cuda = kwargs.get('cuda', False)

        if layer_type == 'lstm':
            self.layers = [LSTMCell(config['input_size'], config['output_size']) for config in layers_config]
        else:
            self.layers = [Layer(config['input_size'], config['output_size'], config['activation']) for config in layers_config]

        if use_cuda:
            if layer_type != 'lstm':
                self.layers = [GPULayer(config['input_size'], config['output_size'], config['activation']) for config in layers_config]
            else:
                raise NotImplementedError("LSTM is not supported on GPU yet.")

        if optimizer_params is None:
            optimizer_params = {'lr': 0.01}

        self.optimizer = self._create_optimizer(optimizer_name, optimizer_params)

    def _create_optimizer(self, optimizer_name, optimizer_params):
        name = optimizer_name.lower()
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
        if isinstance(self.layers[0], LSTMCell):
            batch_size = x.shape[1]
            h, c = np.zeros((self.layers[0].hidden_size, batch_size)), np.zeros((self.layers[0].hidden_size, batch_size))
            for layer in self.layers:
                h, c = layer.forward(x, h, c)
            return h
        else:
            for layer in self.layers:
                x = layer.forward(x)
            return x

    def backward(self, dh):
        dc = np.zeros_like(dh)
        for layer in reversed(self.layers):
            if isinstance(layer, LSTMCell):
                dx, dh, dc = layer.backward(dh, dc)
            else:
                dh = layer.backward(dh)

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            layer.weights = self.optimizer.update(layer.weights, layer.grad_weights, f'layer_{i}_weights')
            layer.biases = self.optimizer.update(layer.biases, layer.grad_biases, f'layer_{i}_biases')

    def train(self, inputs, targets, cuda=False):
        if not cuda:
            outputs = self.forward(inputs)
            loss = ((outputs - targets) ** 2).mean()
            grad_outputs = 2 * (outputs - targets) / outputs.size
            self.backward(grad_outputs)
            self.update_weights()
            return loss
        else:
            return self._train_gpu(inputs, targets)

    def _train_gpu(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = gpu_mse_loss(outputs, targets, outputs.size)
        grad_outputs = 2 * (outputs - targets) / outputs.size
        self.backward(grad_outputs)
        self.update_weights()
        return loss
