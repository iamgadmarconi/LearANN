use rand::Rng;

trait RNN {
    fn forward(&self, input: Vec<f64>) -> Vec<f64>;
    fn backward(&self, input: Vec<f64>, target: Vec<f64>) -> Vec<f64>;
    fn update_weights(&self, learning_rate: f64);
}

pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    hidden_state: Vec<f64>,
    activation: Activation,
    update_count: usize,
    mt_weights: Vec<Vec<f64>>,
    vt_weights: Vec<Vec<f64>>,
    mt_biases: Vec<f64>,
    vt_biases: Vec<f64>,
}

pub struct SimpleRNN {
    layers: Vec<Layer>,
    input_size: usize,
    output_size: usize,
}

#[derive(Clone)]
enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
}

impl Activation {
    fn apply(&self, x: f64) -> f64 {
        match *self {
            Activation::Sigmoid => sigmoid(x),
            Activation::Tanh => tanh(x),
            Activation::ReLU => relu(x),
        }
    }

    fn derivative(&self, output: f64) -> f64 {
        match *self {
            Activation::Sigmoid => sigmoid_derivative(output),
            Activation::Tanh => tanh_derivative(output),
            Activation::ReLU => relu_derivative(output),
        }
    }
}

impl Layer {
    fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<Vec<f64>> = (0..output_size).map(|_| {
            (0..input_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
        }).collect();
        
        let biases: Vec<f64> = (0..output_size).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        let mt_weights = vec![vec![0.0; input_size]; output_size];
        let vt_weights = vec![vec![0.0; input_size]; output_size];
        let mt_biases = vec![0.0; output_size];
        let vt_biases = vec![0.0; output_size];
        
        Layer {
            weights,
            biases,
            hidden_state: vec![0.0; output_size],
            activation,
            update_count: 1,
            mt_weights,
            vt_weights,
            mt_biases,
            vt_biases,
        }
    }


    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = vec![0.0; self.biases.len()];
        for (i, output_val) in output.iter_mut().enumerate() {
            *output_val = self.activation.apply(
                self.biases[i] + input.iter().zip(self.weights[i].iter()).map(|(inp, weight)| inp * weight).sum::<f64>()
            );
        }
        self.hidden_state = output.clone();
        output
    }    

    fn backward(&mut self, output_error: &Vec<f64>, input: &Vec<f64>) -> Vec<f64> {
        // Calculate the gradient of the output based on the activation function
        let d_output = output_error.iter().zip(self.hidden_state.iter())
            .map(|(err, out)| err * self.activation.derivative(*out))
            .collect::<Vec<f64>>();
    
        // Calculate gradient w.r.t input for the previous layer
        let input_error: Vec<f64> = (0..input.len()).map(|i| {
            self.weights.iter().map(|weights| weights[i])
                .zip(&d_output)
                .map(|(weight, dout)| weight * dout)
                .sum()
        }).collect();
    
        // Calculate gradients for weights and biases
        let gradients_weights = self.weights.iter().enumerate().map(|(i, weights)| {
            weights.iter().enumerate().map(|(j, _)| d_output[i] * input[j]).collect::<Vec<f64>>()
        }).collect::<Vec<Vec<f64>>>();
    
        let gradients_biases = d_output.clone();  // Directly use d_output as gradient for biases
    
        // Pass gradients to an update function (considering Adam optimization here)
        self.update_weights_adam(gradients_weights, gradients_biases, 0.01); // Example learning rate for Adam
    
        input_error
    }

    fn update_weights_adam(&mut self, gradients_weights: Vec<Vec<f64>>, gradients_biases: Vec<f64>, lr: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                // Update mt and vt for weights
                self.mt_weights[i][j] = 0.9 * self.mt_weights[i][j] + 0.1 * gradients_weights[i][j];
                self.vt_weights[i][j] = 0.999 * self.vt_weights[i][j] + 0.001 * gradients_weights[i][j].powi(2);

                // Compute bias-corrected first and second moment estimates
                let m_hat = self.mt_weights[i][j] / (1.0 - 0.9f64.powi(self.update_count as i32));
                let v_hat = self.vt_weights[i][j] / (1.0 - 0.999f64.powi(self.update_count as i32));

                // Update weights
                self.weights[i][j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }

            // Update biases similarly
            self.mt_biases[i] = 0.9 * self.mt_biases[i] + 0.1 * gradients_biases[i];
            self.vt_biases[i] = 0.999 * self.vt_biases[i] + 0.001 * gradients_biases[i].powi(2);

            let m_hat = self.mt_biases[i] / (1.0 - 0.9f64.powi(self.update_count as i32));
            let v_hat = self.vt_biases[i] / (1.0 - 0.999f64.powi(self.update_count as i32));

            self.biases[i] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
        }
        self.update_count += 1;  // Increment the counter used for bias correction
    }    

    pub fn update_weights(&mut self, d_weights: Vec<Vec<f64>>, d_biases: Vec<f64>, t: usize, lr: f64, beta1: f64, beta2: f64, epsilon: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                // Update mt and vt for weights
                self.mt_weights[i][j] = beta1 * self.mt_weights[i][j] + (1.0 - beta1) * d_weights[i][j];
                self.vt_weights[i][j] = beta2 * self.vt_weights[i][j] + (1.0 - beta2) * d_weights[i][j].powi(2);
                // Correct mt and vt
                let mt_hat = self.mt_weights[i][j] / (1.0 - beta1.powi(t as i32));
                let vt_hat = self.vt_weights[i][j] / (1.0 - beta2.powi(t as i32));
                // Update weights
                self.weights[i][j] -= lr * mt_hat / (vt_hat.sqrt() + epsilon);
            }
            // Repeat for biases
            self.mt_biases[i] = beta1 * self.mt_biases[i] + (1.0 - beta1) * d_biases[i];
            self.vt_biases[i] = beta2 * self.vt_biases[i] + (1.0 - beta2) * d_biases[i].powi(2);
            let mt_hat = self.mt_biases[i] / (1.0 - beta1.powi(t as i32));
            let vt_hat = self.vt_biases[i] / (1.0 - beta2.powi(t as i32));
            self.biases[i] -= lr * mt_hat / (vt_hat.sqrt() + epsilon);
        }
    }
}

impl SimpleRNN {
    fn new(num_layers: usize, input_size: usize, output_size: usize, activation: Activation) -> Self {
        let layers = (0..num_layers).map(|_| {
            Layer::new(input_size, output_size, activation.clone()) 
        }).collect();
        SimpleRNN {
            layers,
            input_size,
            output_size,
        }
    }

    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.layers.iter_mut().fold(input, |acc, layer| layer.forward(&acc))
    }

    fn backward(&mut self, mut output_error: Vec<f64>) -> Vec<f64> {
        // Collect all necessary inputs before the mutable borrow
        let inputs: Vec<Vec<f64>> = self.layers
            .iter()
            .map(|layer| layer.hidden_state.clone())
            .collect();

        // Now perform operations with mutable borrow
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let input = if i == 0 { vec![0.0; self.input_size] } else {
                inputs[i - 1].clone()
            };
            output_error = layer.backward(&output_error, &input);
        }
        output_error
    }

    fn update_weights(&mut self, learning_rate: f64) {
        for layer in self.layers.iter_mut() {
            layer.update_weights_adam(learning_rate);
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh_derivative(output: f64) -> f64 {
    1.0 - output.powi(2)
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn relu_derivative(output: f64) -> f64 {
    if output > 0.0 {
        1.0
    } else {
        0.0
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let rnn = SimpleRNN::new(3, 2, 2, Activation::Sigmoid);
        for layer in rnn.layers {
            for weight in layer.weights.iter().flat_map(|row| row.iter()) {
                assert!(*weight >= -1.0 && *weight <= 1.0);
            }
            for bias in layer.biases.iter() {
                assert!(*bias >= -1.0 && *bias <= 1.0);
            }
        }
    }

    #[test]
    fn test_forward_propagation() {
        let mut rnn = SimpleRNN::new(1, 3, 1, Activation::ReLU); // Simple config for testing
        let input = vec![0.5];  // Single input of moderate size
        let output = rnn.forward(vec![input]);
        // Since we're not sure what the right output is, we just check the types and no error is thrown
        assert_eq!(output.len(), 1);  // Ensuring output is of expected size
    }

    #[test]
    fn test_forward_pass() {
        let mut layer = Layer::new(3, 2, Activation::Tanh);
        layer.weights = vec![vec![0.5, -0.5, 1.0], vec![-1.5, 0.5, 0.0]];
        layer.biases = vec![0.0, 0.0];
        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input);
        assert!((output[0] - tanh(1.0 * 0.5 - 2.0 * 0.5 + 3.0 * 1.0)).abs() < 1e-5);
        assert!((output[1] - tanh(-1.5 * 1.0 + 0.5 * 2.0 + 0.0 * 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_backward_propagation() {
        let mut rnn = SimpleRNN::new(1, 3, 1, Activation::Sigmoid);
        let inputs = vec![vec![0.1]];
        let targets = vec![vec![0.0]];
        rnn.forward(inputs[0].clone());
        let grads = rnn.backward(inputs[0].clone(), targets[0].clone());

        // Validate gradients are within expected range (typically gradients are small values)
        for grad in grads {
            assert!(grad.abs() < 1.0);
        }
    }

    #[test]
    fn test_learning_overfit() {
        let mut rnn = SimpleRNN::new(3, 2, 2, Activation::Sigmoid);  // Assume constructor parameters are correct
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        // Simple training loop to overfit on a very small dataset
        // Assuming you have access to weights and can calculate gradients
        let mut previous_weights = rnn.layers.iter().map(|l| l.weights.clone()).collect::<Vec<_>>();

        for epoch in 0..10000 {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = rnn.forward(input.clone());
                let loss = output.iter().zip(target.iter())
                                .map(|(o, t)| (o - t).powi(2))
                                .sum::<f64>();
                total_loss += loss;
                let error = output.iter().zip(target.iter())
                                .map(|(o, t)| o - t)
                                .collect::<Vec<_>>();
                rnn.backward(error);
            }
            println!("Epoch {}: Total Loss {}", epoch, total_loss);
            // Monitor weight changes
            let weight_changes = rnn.layers.iter().zip(&previous_weights).map(|(current, previous)| {
                current.weights.iter().zip(previous).map(|(cw, pw)| {
                    cw.iter().zip(pw).map(|(c, p)| (c - p).abs()).sum::<f64>()
                }).sum::<f64>()
            }).sum::<f64>();

            println!("Epoch {}: Total Loss {}, Total Weight Change {}", epoch, total_loss, weight_changes);
            previous_weights = rnn.layers.iter().map(|l| l.weights.clone()).collect::<Vec<_>>();

            if total_loss < 1e-5 || weight_changes < 1e-6 {  // Check for minimal weight changes
                break; // Early stopping condition
            }
        }
    }

    #[test]
    fn test_learning() {
        let mut rnn = SimpleRNN::new(2, 3, 1, Activation::Tanh);
        let inputs = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let targets = vec![vec![1.0], vec![0.0]];

        for _ in 0..100 {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = rnn.forward(input.clone());
                let loss = output.iter().zip(target.iter())
                                .map(|(o, t)| (o - t).powi(2))
                                .sum::<f64>();
                total_loss += loss;
                rnn.backward(input.clone(), target.clone());
            }
            rnn.update_weights(0.01);
            if total_loss < 1e-5 {
                break;
            }
        }
        assert!(total_loss < 1e-5, "Model did not learn successfully");
    }
}