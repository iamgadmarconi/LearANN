use ndarray::{Array2, Axis, concatenate, stack, s, arr2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::any::Any;

use super::Layer;

pub struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    weights: Array2<f64>,
    biases: Array2<f64>,
    input: Option<Array2<f64>>,
    h_prev: Option<Array2<f64>>,
    c_prev: Option<Array2<f64>>,
    i_gate: Option<Array2<f64>>,
    f_gate: Option<Array2<f64>>,
    o_gate: Option<Array2<f64>>,
    g_gate: Option<Array2<f64>>,
    c: Option<Array2<f64>>,
    grad_weights: Option<Array2<f64>>,
    grad_biases: Option<Array2<f64>>,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weights = Array2::random((4 * hidden_size, input_size + hidden_size), Uniform::new(-1.0, 1.0)) * (2.0 / (input_size + hidden_size) as f64).sqrt();
        let biases = Array2::zeros((4 * hidden_size, 1));

        LSTMCell {
            input_size,
            hidden_size,
            weights,
            biases,
            input: None,
            h_prev: None,
            c_prev: None,
            i_gate: None,
            f_gate: None,
            o_gate: None,
            g_gate: None,
            c: None,
            grad_weights: None,
            grad_biases: None,
        }
    }

    fn resize_array2(arr: &Array2<f64>, new_shape: (usize, usize)) -> Array2<f64> {
        let total_elements = arr.len();
        let mut resized_vec = Vec::with_capacity(new_shape.0 * new_shape.1);

        for i in 0..new_shape.0 {
            for j in 0..new_shape.1 {
                let index = (i * new_shape.1 + j) % total_elements;
                let row = index / arr.ncols();
                let col = index % arr.ncols();
                resized_vec.push(arr[[row, col]]);
            }
        }

        Array2::from_shape_vec(new_shape, resized_vec).unwrap()
    }

    pub fn forward(&mut self, x: &Array2<f64>, h_prev: &Array2<f64>, c_prev: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let combined = concatenate![Axis(0), h_prev.view(), x.view()];
        let gates = self.weights.dot(&combined) + &self.biases;
    
        let i_gate = gates.slice(s![..self.hidden_size, ..]).mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let f_gate = gates.slice(s![self.hidden_size..2 * self.hidden_size, ..]).mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let o_gate = gates.slice(s![2 * self.hidden_size..3 * self.hidden_size, ..]).mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let g_gate = gates.slice(s![3 * self.hidden_size.., ..]).mapv(|v| v.tanh());
    
        let c = &f_gate * c_prev + &i_gate * &g_gate;
        let h = &o_gate * c.mapv(|v| v.tanh());
    
        self.input = Some(combined);
        self.h_prev = Some(h_prev.clone());
        self.c_prev = Some(c_prev.clone());
        self.i_gate = Some(i_gate.clone());
        self.f_gate = Some(f_gate.clone());
        self.o_gate = Some(o_gate.clone());
        self.g_gate = Some(g_gate.clone());
        self.c = Some(c.clone());
    
        (h, c)
    }

    pub fn backward(&mut self, dh_next: &Array2<f64>, dc_next: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let batch_size = dh_next.ncols();
        let mut dh_next = dh_next.clone();
        let mut dc_next = dc_next.clone();

        // Adjust dh_next and dc_next if their dimensions do not match self.hidden_size
        if dh_next.nrows() != self.hidden_size {
            dh_next = Self::resize_array2(&dh_next, (self.hidden_size, batch_size));
        }
        if dc_next.nrows() != self.hidden_size {
            dc_next = Self::resize_array2(&dc_next, (self.hidden_size, batch_size));
        }

        let do_gate = dh_next.clone() * self.c.as_ref().unwrap().mapv(|v| v.tanh()) * self.o_gate.as_ref().unwrap().mapv(|v| v * (1.0 - v));
        let dc = dc_next + dh_next.clone() * self.o_gate.as_ref().unwrap() * self.c.as_ref().unwrap().mapv(|v| 1.0 - v.tanh().powi(2));

        let di_gate = dc.clone() * self.g_gate.as_ref().unwrap() * self.i_gate.as_ref().unwrap().mapv(|v| v * (1.0 - v));
        let df_gate = dc.clone() * self.c_prev.as_ref().unwrap() * self.f_gate.as_ref().unwrap().mapv(|v| v * (1.0 - v));
        let dg_gate = dc.clone() * self.i_gate.as_ref().unwrap() * self.g_gate.as_ref().unwrap().mapv(|v| 1.0 - v.powi(2));

        let d_combined = stack![
            Axis(0),
            di_gate,
            df_gate,
            do_gate,
            dg_gate
        ];

        let d_combined = d_combined.into_shape((4 * self.hidden_size, batch_size)).unwrap();
        self.grad_weights = Some(d_combined.dot(&self.input.as_ref().unwrap().t()));
        self.grad_biases = Some(d_combined.sum_axis(Axis(1)).insert_axis(Axis(1)));

        let d_combined_input = self.weights.t().dot(&d_combined);
        let dx = d_combined_input.slice(s![self.hidden_size.., ..]).to_owned();
        let dh_prev = d_combined_input.slice(s![..self.hidden_size, ..]).to_owned();
        let dc_prev = dc * self.f_gate.as_ref().unwrap();

        (dx, dh_prev, dc_prev)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }    
}

impl Layer for LSTMCell {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward(&mut self, _x: &Array2<f64>) -> Array2<f64> {
        unimplemented!()
    }

    fn backward(&mut self, _grad_output: &Array2<f64>) -> Array2<f64> {
        unimplemented!()
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
        self.hidden_size
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
