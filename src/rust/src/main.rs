extern crate ndarray;
extern crate plotters;
extern crate rand;

mod layers;
mod optimizer;
mod rnn;

use rnn::{RNN, LayerConfig};
use ndarray::prelude::*;
use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::collections::HashMap;
use plotters::prelude::*;
use rand::Rng;

fn generate_sine_wave_data(seq_length: usize, num_sequences: usize) -> (Array2<f64>, Array2<f64>) {
    let mut rng = rand::thread_rng();
    let mut X = Array2::<f64>::zeros((num_sequences, seq_length));
    let mut y = Array2::<f64>::zeros((num_sequences, seq_length));

    for i in 0..num_sequences {
        let start = rng.gen_range(0.0..std::f64::consts::PI);
        for j in 0..seq_length {
            X[[i, j]] = (start + j as f64).sin();
            y[[i, j]] = (start + j as f64 + 1.0).sin();
        }
    }

    (X, y)
}

// Function to plot the results
fn plot_sine_wave(y_test: &Array2<f64>, predictions: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("sine_wave.png", (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let (upper, lower) = root_area.split_vertically(512);

    let y_test_row = y_test.index_axis(Axis(0), 0);
    let predictions_row = predictions.index_axis(Axis(0), 0);

    let mut chart = ChartBuilder::on(&upper)
        .caption("Sine Wave - Actual vs Predicted", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..y_test_row.len(), -1.0..1.0)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        (0..y_test_row.len()).map(|x| (x, y_test_row[x])),
        &BLUE,
    ))?
    .label("Actual")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        (0..predictions_row.len()).map(|x| (x, predictions_row[x])),
        &RED,
    ))?
    .label("Predicted")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn main() {
    // Generate sine wave data
    let seq_length = 50;
    let num_sequences = 1000;
    let (X, y) = generate_sine_wave_data(seq_length, num_sequences);
    let X_train = X.slice(s![..800, ..]).to_owned();
    let y_train = y.slice(s![..800, ..]).to_owned();
    let X_test = X.slice(s![800.., ..]).to_owned();
    let y_test = y.slice(s![800.., ..]).to_owned();

    // Define layer configurations
    let layer_configs = vec![
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: seq_length,
            output_size: 50,
            activation: Some("tanh".to_string()),
        },
        LayerConfig {
            layer_type: "lstm".to_string(),
            input_size: 50,
            output_size: 100,
            activation: Some("relu".to_string()),
        },
        LayerConfig {
            layer_type: "lstm".to_string(),
            input_size: 100,
            output_size: 200,
            activation: Some("tanh".to_string()),
        },
        LayerConfig {
            layer_type: "lstm".to_string(),
            input_size: 200,
            output_size: 100,
            activation: Some("tanh".to_string()),
        },
        LayerConfig {
            layer_type: "lstm".to_string(),
            input_size: 100,
            output_size: 50,
            activation: Some("relu".to_string()),
        },
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: 50,
            output_size: seq_length,
            activation: Some("tanh".to_string()),
        },
    ];

    // Initialize optimizer parameters
    let mut optimizer_params = HashMap::new();
    optimizer_params.insert("lr".to_string(), 1e-4);

    // Create RNN
    let mut rnn = RNN::new(layer_configs, "adam", Some(optimizer_params), false, "mse");

    // Training loop
    for epoch in 0..100 {
        let mut total_loss = 0.0;
        for i in 0..X_train.shape()[0] {
            let input = X_train.slice(s![i, ..]).to_owned().insert_axis(Axis(1));
            let target = y_train.slice(s![i, ..]).to_owned().insert_axis(Axis(1));
            let loss = rnn.train(input, target, true);
            total_loss += loss;
        }
        if epoch % 10 == 0 {
            println!("Epoch {}, Loss: {}", epoch, total_loss / X_train.shape()[0] as f64);
        }
    }

    // Testing loop
    let mut predictions = Vec::new();
    for i in 0..X_test.shape()[0] {
        let input = X_test.slice(s![i, ..]).to_owned().insert_axis(Axis(1));
        let prediction = rnn.forward(input);
        predictions.push(prediction.into_raw_vec());
    }

    let predictions = Array2::from_shape_vec((X_test.shape()[0], seq_length), predictions.concat())
        .expect("Error reshaping predictions array");

    // Plot the results
    plot_sine_wave(&y_test, &predictions).expect("Plotting failed");
}