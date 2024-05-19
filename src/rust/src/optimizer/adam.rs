use ndarray::prelude::*;
use std::collections::HashMap;

use super::Optimizer;

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: HashMap<String, Array2<f64>>,
    v: HashMap<String, Array2<f64>>,
    t: usize,
    param_index: HashMap<String, usize>,
    param_shapes: Vec<(usize, usize)>,
    initialized: bool,
}

impl Adam {
    pub fn new(params: &HashMap<String, f64>) -> Self {
        Adam {
            learning_rate: *params.get("lr").unwrap_or(&0.001),
            beta1: *params.get("beta1").unwrap_or(&0.9),
            beta2: *params.get("beta2").unwrap_or(&0.999),
            epsilon: *params.get("epsilon").unwrap_or(&1e-8),
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            param_index: HashMap::new(),
            param_shapes: Vec::new(),
            initialized: false,
        }
    }

    pub fn initialize_parameters(&mut self, param_shapes: &HashMap<String, (usize, usize)>) {
        if !self.initialized {
            self.initialized = true;

            for (i, (name, shape)) in param_shapes.iter().enumerate() {
                self.param_index.insert(name.clone(), i);
                self.param_shapes.push(*shape);
                self.m.insert(name.clone(), Array2::zeros(*shape));
                self.v.insert(name.clone(), Array2::zeros(*shape));
            }
        }
    }

    pub fn update(&mut self, param: &Array2<f64>, grad_param: &Array2<f64>, name: &str) -> Array2<f64> {
        if !self.initialized {
            panic!("Optimizer parameters not initialized. Call `initialize_parameters` first.");
        }

        let idx = self.param_index.get(name).expect("Parameter name not found in index.");
        let mut param = param.to_owned();
        let grad_param = grad_param.to_owned();

        self.t += 1;
        let m = self.m.get_mut(name).unwrap();
        let v = self.v.get_mut(name).unwrap();

        *m = self.beta1 * m + (1.0 - self.beta1) * &grad_param;
        *v = self.beta2 * v + (1.0 - self.beta2) * grad_param.mapv(|g| g * g);

        let m_hat = m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = v.mapv(|v| v.sqrt()) / (1.0 - self.beta2.powi(self.t as i32));

        param -= &(self.learning_rate * &m_hat / (v_hat + self.epsilon));
        param
    }
}

impl Optimizer for Adam {
    fn update(&mut self, param: &Array2<f64>, grad: &Array2<f64>, param_name: &str) -> Array2<f64> {
        self.update(param, grad, param_name)
    }
}
