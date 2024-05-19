use ndarray::prelude::*;
use std::collections::HashMap;

use crate::layers::{Layer, dense::DenseLayer, lstm::LSTMCell};
use crate::optimizer::{Optimizer, adam::Adam, gradient_descent::GradientDescent};

pub struct RNN {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
    loss_function: LossFunction,
}

impl RNN {
    pub fn new(layers_config: Vec<LayerConfig>, optimizer_name: &str, optimizer_params: Option<HashMap<String, f64>>, cuda: bool, loss_function_name: &str) -> Self {
        let loss_function = match loss_function_name {
            "mse" => LossFunction::MSE,
            "crossentropy" => LossFunction::CrossEntropy,
            _ => panic!("Unsupported loss function: {}", loss_function_name),
        };

        let mut layers: Vec<Box<dyn Layer>> = Vec::new();

        for config in layers_config {
            match config.layer_type.as_str() {
                "lstm" => {
                    if cuda {
                        panic!("LSTM not implemented for GPU");
                    }
                    println!("Initializing LSTMCell with input_size={} and hidden_size={}", config.input_size, config.output_size);
                    layers.push(Box::new(LSTMCell::new(config.input_size, config.output_size)));
                }
                _ => {
                    if cuda {
                        println!("Initializing GPU Layers with input_size={} and output_size={}", config.input_size, config.output_size);
                        // Replace GPULayer with the correct struct if it's implemented
                        // layers.push(Box::new(GPULayer::new(config.input_size, config.output_size, config.activation.clone())));
                    } else {
                        println!("Initializing Layer with input_size={} and output_size={}", config.input_size, config.output_size);
                        layers.push(Box::new(DenseLayer::new(config.input_size, config.output_size, config.activation.clone())));
                    }
                }
            }
        }

        let optimizer_params = optimizer_params.unwrap_or_else(|| {
            let mut default_params = HashMap::new();
            default_params.insert("lr".to_string(), 0.01);
            default_params
        });

        let optimizer = RNN::create_optimizer(optimizer_name, &optimizer_params);

        RNN {
            layers,
            optimizer,
            loss_function,
        }
    }

    fn create_optimizer(name: &str, params: &HashMap<String, f64>) -> Box<dyn Optimizer> {
        match name.to_lowercase().as_str() {
            "gradientdescent" => Box::new(GradientDescent::new(params)),
            "adam" => {
                let mut optimizer = Adam::new(params);
                optimizer.initialize_parameters();
                Box::new(optimizer)
            }
            _ => panic!("Unsupported optimizer: {}", name),
        }
    }

    pub fn forward(&self, x: Array2<f64>) -> Array2<f64> {
        let mut x = x;
        let mut hidden_states: Vec<(Array2<f64>, Array2<f64>)> = Vec::new();

        for layer in &self.layers {
            if let Some(lstm_layer) = layer.as_any().downcast_ref::<LSTMCell>() {
                let h = Array2::zeros((lstm_layer.hidden_size, x.ncols()));
                let c = Array2::zeros((lstm_layer.hidden_size, x.ncols()));
                hidden_states.push((h, c));
            }
        }

        let mut h_c_index = 0;
        for layer in &self.layers {
            if let Some(lstm_layer) = layer.as_any().downcast_ref::<LSTMCell>() {
                let (mut h, mut c) = hidden_states[h_c_index].clone();
                let (new_h, new_c) = lstm_layer.forward(&x, &mut h, &mut c);
                hidden_states[h_c_index] = (new_h.clone(), new_c.clone());
                x = new_h;
                h_c_index += 1;
            } else {
                x = layer.forward(&x);
            }
        }

        x
    }

    pub fn backward(&mut self, dh: Array2<f64>) {
        let mut dh = dh;
        let mut dc = Array2::zeros(dh.dim()); // Only needed if the last layer is LSTM

        for (i, layer) in self.layers.iter().enumerate().rev() {
            if let Some(lstm_layer) = layer.as_any().downcast_ref::<LSTMCell>() {
                let (new_dh, new_dc) = lstm_layer.backward(&dh, &dc);
                dh = new_dh;
                dc = new_dc;
            } else {
                if dh.nrows() != layer.output_size() {
                    dh = dh.resize((layer.output_size(), dh.ncols()), 0.0);
                }
                dh = layer.backward(&dh);
            }
        }
    }

    pub fn update_weights(&mut self) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.update_weights(&mut *self.optimizer, i);
        }
    }

    pub fn train(&mut self, inputs: Array2<f64>, targets: Array2<f64>, cuda: bool) -> f64 {
        if !cuda {
            let outputs = self.forward(inputs);
            let loss = self.loss(outputs.clone(), targets.clone());
            let grad_outputs = self.loss_grad(outputs, targets);
            self.backward(grad_outputs);
            self.update_weights();
            loss
        } else {
            self.train_gpu(inputs, targets)
        }
    }

    fn train_gpu(&mut self, inputs: Array2<f64>, targets: Array2<f64>) -> f64 {
        let outputs = self.forward(inputs);
        let loss = gpu_mse_loss(outputs.clone(), targets.clone());
        let grad_outputs = 2.0 * (outputs - targets) / outputs.len_of(Axis(1)) as f64;
        self.backward(grad_outputs);
        self.update_weights();
        loss
    }

    fn loss(&self, outputs: Array2<f64>, targets: Array2<f64>) -> f64 {
        match self.loss_function {
            LossFunction::MSE => mean_squared_error(outputs, targets),
            LossFunction::CrossEntropy => cross_entropy_loss(outputs, targets),
        }
    }

    fn loss_grad(&self, outputs: Array2<f64>, targets: Array2<f64>) -> Array2<f64> {
        match self.loss_function {
            LossFunction::MSE => mse_grad(outputs, targets),
            LossFunction::CrossEntropy => cross_entropy_grad(outputs, targets),
        }
    }

    pub fn predict(&self, x: Array2<f64>) -> Array2<f64> {
        let mut predictions: Vec<Array1<f64>> = Vec::new();

        for row in x.genrows() {
            let input = row.to_owned().insert_axis(Axis(1));
            let prediction = self.forward(input);
            predictions.push(prediction.into_shape(prediction.len()).unwrap());
        }

        stack(Axis(0), &predictions.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }
}

#[derive(Clone)]
pub struct LayerConfig {
    pub layer_type: String,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Option<String>,
}

pub enum LossFunction {
    MSE,
    CrossEntropy,
}

pub fn mean_squared_error(outputs: Array2<f64>, targets: Array2<f64>) -> f64 {
    ((&outputs - &targets).mapv(|x| x.powi(2)).sum()) / outputs.len_of(Axis(1)) as f64
}

pub fn mse_grad(outputs: Array2<f64>, targets: Array2<f64>) -> Array2<f64> {
    2.0 * (outputs - targets) / outputs.len_of(Axis(1)) as f64
}

pub fn cross_entropy_loss(outputs: Array2<f64>, targets: Array2<f64>) -> f64 {
    -(&targets * outputs.mapv(|x| x.ln())).sum() / outputs.len_of(Axis(1)) as f64
}

pub fn cross_entropy_grad(outputs: Array2<f64>, targets: Array2<f64>) -> Array2<f64> {
    outputs - targets
}

pub fn gpu_mse_loss(outputs: Array2<f64>, targets: Array2<f64>) -> f64 {
    mean_squared_error(outputs, targets) // Placeholder for actual GPU implementation
}
