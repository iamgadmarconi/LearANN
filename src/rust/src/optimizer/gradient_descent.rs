use ndarray::prelude::*;
use std::collections::HashMap;

use super::Optimizer;

pub struct GradientDescent {
    lr: f64,
}

impl GradientDescent {
    pub fn new(params: &HashMap<String, f64>) -> Self {
        let lr = *params.get("lr").unwrap_or(&0.01);
        GradientDescent { lr }
    }
}

impl Optimizer for GradientDescent {
    fn update(&mut self, param: &Array2<f64>, grad: &Array2<f64>, _param_name: &str) -> Array2<f64> {
        param - &(self.lr * grad)
    }
}
