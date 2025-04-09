#[cfg(test)]
mod tests {
    use kerasty::{Dense, Device, Loss, Metric, Model, Optimizer, Result, Sequential, Tensor};

    #[test]
    fn test_xor_problem() -> Result<()> {
        // Define the XOR input and output data
        let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let x = Tensor::from_slice(&x_data, (4, 2), &Device::Cpu)?;
        let y_data = vec![0.0, 1.0, 1.0, 0.0];
        let y = Tensor::from_slice(&y_data, (4, 1), &Device::Cpu)?;

        // Build the neural network model
        let mut model = Sequential::new();
        model.add(Dense::new(2, 2, "relu"));
        model.add(Dense::new(1, 2, "sigmoid"));

        println!("Model built successfully");

        // Compile the model
        model.compile(
            Optimizer::Adam(0.001, 0.9, 0.999, 1e-8, 0.0),
            Loss::BinaryCrossEntropyWithLogit,
            vec![Metric::Accuracy],
        )?;

        println!("Model compiled successfully");

        // Train the model
        model.fit(x.clone(), y.clone(), 100)?;

        // Make predictions
        let predictions = model.predict(&x);
        let predictions = predictions.reshape(4)?.to_vec1::<f64>()?;
        let predictions: Vec<i32> = predictions
            .iter()
            .map(|&p| if p >= 0.5 { 1 } else { 0 })
            .collect();

        println!("Predictions:");
        for i in 0..4 {
            println!(
                "Input: {:?} => Predicted Output: {}, Actual Output: {}",
                &x_data[i * 2..i * 2 + 2],
                predictions[i],
                y_data[i]
            );
        }

        // Evaluate the model
        let score = model.evaluate(&x, &y);
        println!("Average loss: {}", score[0]);
        println!("Accuracy: {}", score[1]);

        Ok(())
    }
}
