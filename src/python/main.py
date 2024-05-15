from networks.rnn.tests.test_rnn import test_rnn, test_rnn_on_sine_wave, test_rnn_with_adagrad, test_rnn_with_adam, test_rnn_with_grad_descent, test_rnn_with_cuda


if __name__ == '__main__':
    # test_rnn()
    # test_rnn_on_sine_wave()
    # test_rnn_with_adagrad()
    # test_rnn_with_adam()
    # test_rnn_with_grad_descent()

    test_rnn_with_cuda()

    print("Done!")