use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::any::Any;

use super::Layer;

pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Array2<f64>,
    biases: Array2<f64>,
    activation: Activation,
    input: Option<Array2<f64>>,
    z: Option<Array2<f64>>,
    grad_weights: Option<Array2<f64>>,
    grad_biases: Option<Array2<f64>>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Option<String>) -> Self {
        let weights = Array2::random((output_size, input_size), Uniform::new(-1.0, 1.0)) * (2.0 / input_size as f64).sqrt();
        let biases = Array2::zeros((output_size, 1));
        let activation = match activation.as_deref() {
            Some("relu") => Activation::ReLU,
            Some("sigmoid") => Activation::Sigmoid,
            Some("tanh") => Activation::Tanh,
            _ => Activation::Linear,
        };

        DenseLayer {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            input: None,
            z: None,
            grad_weights: None,
            grad_biases: None,
        }
    }

    fn activation(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activation {
            Activation::ReLU => z.mapv(|v| if v > 0.0 { v } else { 0.0 }),
            Activation::Sigmoid => z.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => z.mapv(|v| v.tanh()),
            Activation::Linear => z.clone(),
        }
    }

    fn activation_grad(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activation {
            Activation::ReLU => z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => {
                let sig = z.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                sig * (1.0 - sig)
            }
            Activation::Tanh => z.mapv(|v| 1.0 - v.tanh().powi(2)),
            Activation::Linear => Array2::ones(z.raw_dim()),
        }
    }
}

impl Layer for DenseLayer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let x = if x.ndim() == 1 {
            x.clone().insert_axis(Axis(1)).into_dimensionality::<Ix2>().unwrap()
        } else {
            x.clone()
        };
        self.input = Some(x.clone());
        self.z = Some(self.weights.dot(&x) + &self.biases);
        self.activation(&self.z.as_ref().unwrap())
    }    

    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64> {
        let grad_output = if grad_output.ndim() == 1 {
            grad_output.clone().insert_axis(Axis(1)).into_dimensionality::<Ix2>().unwrap()
        } else {
            grad_output.clone()
        };
    
        let grad_z = grad_output * self.activation_grad(&self.z.as_ref().unwrap());
    
        self.grad_weights = Some(grad_z.dot(&self.input.as_ref().unwrap().t()));
        self.grad_biases = Some(grad_z.sum_axis(Axis(1)).insert_axis(Axis(1)));
    
        self.weights.t().dot(&grad_z)
    }    

    fn update_weights(&mut self, optimizer: &mut dyn crate::optimizer::Optimizer, layer_index: usize) {
        self.weights = optimizer.update(
            &self.weights,
            self.grad_weights.as_ref().unwrap(),
            &format!("layer_{}_weights", layer_index),
        );
        self.biases = optimizer.update(
            &self.biases,
            self.grad_biases.as_ref().unwrap(),
            &format!("layer_{}_biases", layer_index),
        );
    }

    fn output_size(&self) -> usize {
        self.output_size
    }
}

enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}
