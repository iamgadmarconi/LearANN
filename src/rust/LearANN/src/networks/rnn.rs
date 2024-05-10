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
}

pub struct SimpleRNN {
    layers: Vec<Layer>,
    input_size: usize,
    output_size: usize,
}

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
        
        Layer {
            weights,
            biases,
            hidden_state: vec![0.0; output_size],
            activation,
        }
    }


    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = vec![0.0; self.biases.len()];
        for (i, output_val) in output.iter_mut().enumerate() {
            *output_val = self.activation.apply(self.biases[i] + input.iter().zip(self.weights[i].iter()).map(|(inp, weight)| inp * weight).sum::<f64>());
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

        // Store gradients to update weights later
        self.update_weights(&d_output, input, 0.01);  // Example learning rate

        input_error
    }

    fn update_weights(&mut self, d_output: &Vec<f64>, input: &Vec<f64>, learning_rate: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                // Update weights using gradient descent
                self.weights[i][j] -= learning_rate * d_output[i] * input[j];
            }
            // Update biases
            self.biases[i] -= learning_rate * d_output[i];
        }
    }
}

impl SimpleRNN {
    fn new(num_layers: usize, input_size: usize, output_size: usize) -> Self {
        let layers = (0..num_layers).map(|_| {
            Layer::new(input_size, output_size, Activation::Sigmoid) // Example activation (change to desired activation)
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
    fn test_rnn() {
        // Initialize an RNN with one layer
        let mut rnn = SimpleRNN {
            layers: vec![
                Layer::new(3, 2, Activation::Sigmoid), // Input size 3, Output size 2
            ],
            input_size: 3,
            output_size: 2,
        };

        // Simulate some input data
        let input = vec![0.5, -1.5, 0.3];
        
        // Forward pass
        let output = rnn.forward(input.clone());
        println!("Output after forward pass: {:?}", output);

        // Dummy target output for loss calculation (for testing)
        let target = vec![0.0, 1.0];
        // Calculate mean squared error and its gradient
        let error: Vec<f64> = output.iter().zip(target.iter())
            .map(|(o, t)| o - t)
            .collect();
        let loss: f64 = error.iter().map(|e| e.powi(2)).sum::<f64>() / error.len() as f64;
        println!("Loss: {}", loss);

        // Calculate gradients for a backward pass (simple derivative of MSE)
        let gradients: Vec<f64> = error.iter().map(|&e| 2.0 * e / error.len() as f64).collect();

        // Backward pass
        rnn.backward(gradients);
        println!("Backward pass completed");
    }
}
