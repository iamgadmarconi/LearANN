pub mod dense;
pub mod lstm;

use ndarray::Array2;

pub trait Layer {
    fn as_any(&self) -> &dyn std::any::Any;
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, grad_output: &Array2<f64>) -> Array2<f64>;
    fn update_weights(&mut self, optimizer: &mut dyn crate::optimizer::Optimizer, layer_index: usize);
    fn output_size(&self) -> usize;
}
