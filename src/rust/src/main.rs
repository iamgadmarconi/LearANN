mod layers;
mod optimizer;
mod rnn;



use rnn::{RNN, LayerConfig};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::collections::HashMap;

fn main() {
    // Define layer configurations
    let layer_configs = vec![
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: 10,
            output_size: 20,
            activation: Some("relu".to_string()),
        },
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: 20,
            output_size: 1,
            activation: Some("sigmoid".to_string()),
        },
    ];

    // Initialize optimizer parameters
    let mut optimizer_params = HashMap::new();
    optimizer_params.insert("lr".to_string(), 0.001);
    optimizer_params.insert("beta1".to_string(), 0.9);
    optimizer_params.insert("beta2".to_string(), 0.999);
    optimizer_params.insert("epsilon".to_string(), 1e-8);

    // Create RNN
    let mut rnn = RNN::new(layer_configs, "adam", Some(optimizer_params), false, "mse");

    // Create dummy inputs and targets
    let inputs = Array2::random((10, 5), Uniform::new(-1.0, 1.0));
    let targets = Array2::random((1, 5), Uniform::new(0.0, 1.0));

    // Train the model
    let loss = rnn.train(inputs, targets, false);
    println!("Training loss: {}", loss);
}
