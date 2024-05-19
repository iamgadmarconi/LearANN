pub mod adam;
pub mod gradient_descent;

use ndarray::Array2;

pub trait Optimizer {
    fn update(&mut self, param: &Array2<f64>, grad: &Array2<f64>, param_name: &str) -> Array2<f64>;
}
