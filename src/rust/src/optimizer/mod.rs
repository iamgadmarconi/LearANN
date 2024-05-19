pub mod adam;
pub mod gradient_descent;

use ndarray::Array2;
use std::collections::HashMap;

pub trait Optimizer {
    fn initialize_parameters(&mut self, param_shapes: &HashMap<String, (usize, usize)>);
    fn update(&mut self, param: &Array2<f64>, grad: &Array2<f64>, param_name: &str) -> Array2<f64>;
}
