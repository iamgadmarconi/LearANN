import numpy as np

from utils.optimizers import Adam, Adagrad, GradientDescent
from utils.layers import Layer, GPULayer, LSTMCell
from utils.cuda.cuda import gpu_mse_loss


class RNN:
    def __init__(self, layers_config, optimizer_name='adam', optimizer_params=None, **kwargs):
        use_cuda = kwargs.get('cuda', False)

        self.layers = []

        for config in layers_config:
            try:
                layer_type = config['type']
            except KeyError:
                layer_type = 'dense'

            if layer_type == 'lstm':
                if use_cuda:
                    raise NotImplementedError("LSTM not implemented for GPU")
                print(f"Initializing LSTMCell with input_size={config['input_size']} and hidden_size={config['output_size']}")
                self.layers.append(LSTMCell(config['input_size'], config['output_size']))
            else:
                if use_cuda:
                    print(f"Initializing GPU Layers with input_size={config['input_size']} and output_size={config['output_size']}")
                    self.layers.append(GPULayer(config['input_size'], config['output_size'], config['activation']))
                else:
                    print(f"Initializing Layer with input_size={config['input_size']} and output_size={config['output_size']}")
                    self.layers.append(Layer(config['input_size'], config['output_size'], config['activation']))


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
        batch_size = x.shape[1]
        hidden_states = [(np.zeros((layer.hidden_size, batch_size)), np.zeros((layer.hidden_size, batch_size))) 
                        for layer in self.layers if isinstance(layer, LSTMCell)]

        h_c_index = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LSTMCell):
                h, c = hidden_states[h_c_index]
                # print(f"Layer {i}: LSTM input shape: {x.shape}")
                h, c = layer.forward(x, h, c)
                hidden_states[h_c_index] = (h, c)
                x = h  # Update x to the output h for the next layer
                # print(f"Layer {i}: LSTM output shape: {x.shape}")
                h_c_index += 1
            else:
                # print(f"Layer {i}: Dense input shape: {x.shape}")
                x = layer.forward(x)
                # print(f"Layer {i}: Dense output shape: {x.shape}")
        return x

    def backward(self, dh):
        dc = np.zeros_like(dh)
        h_c_index = len([layer for layer in self.layers if isinstance(layer, LSTMCell)]) - 1
        for i, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, LSTMCell):
                # print(f"Backward Layer {i}: LSTM dh shape: {dh.shape}, dc shape: {dc.shape}")
                dh, dc = self.adjust_gradients(dh, dc, layer.hidden_size)
                dx, dh, dc = layer.backward(dh, dc)
                # print(f"Backward Layer {i}: LSTM dx shape: {dx.shape}, dh shape: {dh.shape}, dc shape: {dc.shape}")
                h_c_index -= 1
            else:
                # print(f"Backward Layer {i}: Dense dh shape: {dh.shape}")
                dh = layer.backward(dh)
                # print(f"Backward Layer {i}: Dense dh shape after backward: {dh.shape}")

    def adjust_gradients(self, dh, dc, hidden_size):
        # Resize gradients to match the hidden size of the next layer in the backward pass
        if dh.shape[0] != hidden_size:
            dh = np.resize(dh, (hidden_size, dh.shape[1]))
            dc = np.resize(dc, (hidden_size, dc.shape[1]))
        return dh, dc

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