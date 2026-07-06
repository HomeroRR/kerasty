//! End-to-end training test: the classic XOR problem.

use kerasty::{Dense, Device, Loss, Metric, Model, Optimizer, Result, Sequential, Tensor};

#[test]
fn test_xor_problem() -> Result<()> {
    // XOR inputs/targets as f32 (the model's default dtype).
    let x_data: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x = Tensor::from_slice(&x_data, (4, 2), &Device::Cpu)?;
    let y_data: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];
    let y = Tensor::from_slice(&y_data, (4, 1), &Device::Cpu)?;

    // A small hidden layer with a smooth activation solves XOR reliably; the
    // sigmoid output pairs with `BinaryCrossEntropy` (probabilities in) — no
    // double-sigmoid as would happen with the *WithLogit variant.
    let mut model = Sequential::new().with_device(Device::Cpu);
    model.add(Dense::new(8, 2, "tanh"));
    model.add(Dense::new(1, 8, "sigmoid"));

    model.compile(
        Optimizer::adam(0.05),
        Loss::BinaryCrossEntropy,
        vec![Metric::Accuracy],
    )?;

    model.fit(x.clone(), y.clone(), 3_000)?;

    let predictions = model.predict(&x)?;
    let predictions = predictions.reshape(4)?.to_vec1::<f32>()?;
    let predictions: Vec<i32> = predictions
        .iter()
        .map(|&p| if p >= 0.5 { 1 } else { 0 })
        .collect();

    for (i, chunk) in x_data.chunks(2).enumerate() {
        println!(
            "Input: {:?} => Predicted: {}, Actual: {}",
            chunk, predictions[i], y_data[i]
        );
    }

    let score = model.evaluate(&x, &y)?;
    println!("Average loss: {}", score[0]);
    println!("Accuracy: {}", score[1]);

    // XOR is fully learnable; after training we expect perfect accuracy.
    assert!(
        score[1] >= 0.99,
        "expected the model to solve XOR, got accuracy {}",
        score[1]
    );

    Ok(())
}
