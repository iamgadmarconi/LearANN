pub mod dense;
pub mod lstm;

use ndarray::Array2;
use std::any::Any;
use crate::optimizer::Optimizer;

pub trait Layer {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64>;
    fn update_weights(&mut self, optimizer: &mut dyn Optimizer, layer_index: usize);
    fn output_size(&self) -> usize;
}
