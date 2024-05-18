import numpy as np
import matplotlib.pyplot as plt

from networks.rnn.rnn import RNN

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

    # Convert data to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Configure and create RNN
    layers_config = [
        {'input_size': seq_length, 'output_size': 50, 'activation': 'tanh'},
        {'input_size': 50, 'output_size': 100, 'activation': 'relu'},
        {'input_size': 100, 'output_size': 50, 'activation': 'relu'},
        {'input_size': 50, 'output_size': 50, 'activation': 'relu'},
        {'input_size': 50, 'output_size': seq_length, 'activation': 'tanh'}
    ]

    optimizer_config = {'lr': 1e-8}

    rnn = RNN(layers_config, optimizer_name='adam', optimizer_params=optimizer_config)

    # Training loop
    for epoch in range(100):
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


def test_rnn_with_adam():
    np.random.seed(42)  # For reproducibility

    layers_config = [
        {'input_size': 5, 'output_size': 10, 'activation': 'relu'},
        {'input_size': 10, 'output_size': 100, 'activation': 'relu'},
        {'input_size': 100, 'output_size': 1, 'activation': 'linear'}
    ]

    optimizer_config = {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}

    rnn = RNN(layers_config, optimizer_name='adam', optimizer_params=optimizer_config)
    
    # Example training data
    inputs = np.random.randn(5, 1)
    targets = np.random.randn(1, 1)
    
    # Training loop
    for epoch in range(1000):
        loss = rnn.train(inputs, targets)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Test forward pass
    output = rnn.forward(inputs)
    print("Output after training:", output)


def test_rnn_with_adagrad():
    np.random.seed(42)  # For reproducibility

    layers_config = [
        {'input_size': 5, 'output_size': 10, 'activation': 'relu'},
        {'input_size': 10, 'output_size': 100, 'activation': 'relu'},
        {'input_size': 100, 'output_size': 1, 'activation': 'linear'}
    ]

    optimizer_config = {'lr': 0.01, 'epsilon': 1e-4}

    rnn = RNN(layers_config, optimizer_name='adagrad', optimizer_params=optimizer_config)
    
    # Example training data
    inputs = np.random.randn(5, 1)
    targets = np.random.randn(1, 1)
    
    # Training loop
    for epoch in range(1000):
        loss = rnn.train(inputs, targets)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Test forward pass
    output = rnn.forward(inputs)
    print("Output after training:", output)


def test_rnn_with_grad_descent():
    np.random.seed(42)  # For reproducibility

    layers_config = [
        {'input_size': 5, 'output_size': 10, 'activation': 'relu'},
        {'input_size': 10, 'output_size': 100, 'activation': 'relu'},
        {'input_size': 100, 'output_size': 1, 'activation': 'linear'}
    ]

    optimizer_config = {'lr': 0.01}

    rnn = RNN(layers_config, optimizer_name='gradientdescent', optimizer_params=optimizer_config)
    
    # Example training data
    inputs = np.random.randn(5, 1)
    targets = np.random.randn(1, 1)
    
    # Training loop
    for epoch in range(1000):
        loss = rnn.train(inputs, targets)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Test forward pass
    output = rnn.forward(inputs)
    print("Output after training:", output)


def test_rnn_with_cuda():
    np.random.seed(42)  # For reproducibility

    layers_config = [
        {'input_size': 5, 'output_size': 10, 'activation': 'relu'},
        {'input_size': 10, 'output_size': 100, 'activation': 'relu'},
        {'input_size': 100, 'output_size': 1, 'activation': 'linear'}
    ]

    optimizer_config = {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}

    rnn = RNN(layers_config, optimizer_name='adam', optimizer_params=optimizer_config, cuda=True)
    
    # Example training data
    inputs = np.random.randn(5, 1)
    targets = np.random.randn(1, 1)
    
    # Training loop
    for epoch in range(1000):
        loss = rnn.train(inputs, targets)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Test forward pass
    output = rnn.forward(inputs)
    print("Output after training:", output)


def test_rnn_with_cuda_on_sine_wave():
    seq_length = 50
    num_sequences = 1000
    X, y = generate_sine_wave_data(seq_length, num_sequences)
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    # Configure and create RNN
    layers_config = [
        {'input_size': seq_length, 'output_size': 50, 'activation': 'tanh'},
        {'input_size': 50, 'output_size': 100, 'activation': 'relu'},
        {'input_size': 100, 'output_size': 50, 'activation': 'relu'},
        {'input_size': 50, 'output_size': seq_length, 'activation': 'tanh'}
    ]

    optimizer_config = {'lr': 0.01}

    rnn = RNN(layers_config, optimizer_name='adam', optimizer_params=optimizer_config, cuda=True)

    # Training loop
    for epoch in range(100):
        total_loss = 0
        for i in range(len(X_train)):
            loss = rnn.train(X_train[i].reshape(-1, 1), y_train[i].reshape(-1, 1), cuda=True)
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


# def test_rnn_lstm_on_sine_wave():
#     # Generate sine wave data
#     X_train = np.sin(np.linspace(0, 2 * np.pi, 100))
#     y_train = np.roll(X_train, -1)

#     # Define and initialize the model parameters (weights and biases)
#     layers_config = [
#         {'input_size': 1, 'output_size': 50, 'type': 'lstm'},
#         {'input_size': 50, 'output_size': 100, 'type': 'lstm'},
#         {'input_size': 100, 'output_size': 50, 'type': 'lstm'},
#         {'input_size': 50, 'output_size': 1, 'activation': 'linear', 'type': 'dense'}
#     ]

#     # Initialize the RNN model with the Adam optimizer
#     rnn = RNN(layers_config, optimizer_name='adam', optimizer_params={'lr': 0.001}, layer_type='lstm')

#     # Training loop
#     for epoch in range(100):
#         for i in range(len(X_train)):
#             inputs = X_train[i].reshape(1, 1).astype(np.float32)  # Reshape to (input_size, batch_size)
#             targets = y_train[i].reshape(1, 1).astype(np.float32)  # Reshape to (output_size, batch_size)
#             loss = rnn.train(inputs, targets)
#         if epoch % 10 == 0:
#             print(f'Epoch {epoch}, Step {i}, Loss: {loss}')


def test_rnn_lstm_on_sine_wave():
    # Generate sine wave data
    seq_length = 50
    num_sequences = 1000
    X, y = generate_sine_wave_data(seq_length, num_sequences)
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    # Configure and create RNN
    layers_config = [
        {'input_size': seq_length, 'output_size': 50, 'activation': 'tanh'},
        {'input_size': 50, 'output_size': 100, 'activation': 'relu', 'type': 'lstm', 'grad_activation': 'tanh'},
        {'input_size': 100, 'output_size': 200, 'activation': 'tanh', 'type': 'lstm', 'grad_activation': 'tanh'},
        {'input_size': 200, 'output_size': 100, 'activation': 'tanh', 'type': 'lstm', 'grad_activation': 'tanh'},
        {'input_size': 100, 'output_size': 50, 'activation': 'relu', 'type': 'lstm'},
        {'input_size': 50, 'output_size': seq_length, 'activation': 'tanh'}
    ]

    optimizer_config = {'lr': 1e-4}

    rnn = RNN(layers_config, optimizer_name='adam', optimizer_params=optimizer_config)

    # Training loop
    for epoch in range(100):
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