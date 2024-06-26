import cProfile
from networks.rnn.tests.test_rnn import *


if __name__ == '__main__':
    # test_rnn()
    # test_rnn_on_sine_wave()
    # cProfile.run('test_rnn_on_sine_wave()', 'output.prof')
    # test_rnn_with_adagrad()
    # test_rnn_with_adam()
    # test_rnn_with_grad_descent()

    # test_rnn_with_cuda()
    # test_rnn_with_cuda_on_sine_wave()
    # test_rnn_lstm_on_sine_wave()
    cProfile.run('test_rnn_lstm_on_sine_wave()', 'output_lstm.prof')

    print("Done!")