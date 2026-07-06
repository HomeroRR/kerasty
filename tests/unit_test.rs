//! Per-layer shape tests, a mixed-layer CNN, and a save/load round-trip.
//!
//! Each layer is exercised through a `Sequential` model so that `build` and
//! `forward` are covered exactly as they run in production.

use kerasty::{
    AvgPool2D, BatchNorm, Conv1D, Conv2D, Dense, Dropout, Embedding, Flatten,
    GlobalAveragePooling2D, Gru, LayerNorm, Loss, Lstm, MaxPool2D, Model, MultiHeadAttention,
    Optimizer, Reshape, Result, Sequential, SimpleRNN, Tensor,
};
use kerasty::{Device, D};

fn cpu() -> Device {
    Device::Cpu
}

/// Build a one-layer model, run inference, and return the output dims.
fn output_dims<L: kerasty::Layer + 'static>(layer: L, input: &Tensor) -> Result<Vec<usize>> {
    let mut model = Sequential::new().with_device(cpu());
    model.add(layer);
    model.compile(Optimizer::sgd(0.01), Loss::MSE, vec![])?;
    Ok(model.predict(input)?.dims().to_vec())
}

#[test]
fn dense_shape() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (4, 8), &cpu())?;
    assert_eq!(output_dims(Dense::new(3, 8, "relu"), &x)?, vec![4, 3]);
    Ok(())
}

#[test]
fn conv2d_shape() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (2, 1, 8, 8), &cpu())?;
    // valid padding, 3x3 kernel: 8 -> 6
    assert_eq!(
        output_dims(Conv2D::new(1, 4, 3, "relu"), &x)?,
        vec![2, 4, 6, 6]
    );
    Ok(())
}

#[test]
fn conv1d_shape() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (2, 3, 10), &cpu())?;
    assert_eq!(
        output_dims(Conv1D::new(3, 5, 3, "relu"), &x)?,
        vec![2, 5, 8]
    );
    Ok(())
}

#[test]
fn pooling_shapes() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (2, 3, 8, 8), &cpu())?;
    assert_eq!(output_dims(MaxPool2D::new(2), &x)?, vec![2, 3, 4, 4]);
    assert_eq!(output_dims(AvgPool2D::new(2), &x)?, vec![2, 3, 4, 4]);
    assert_eq!(output_dims(GlobalAveragePooling2D::new(), &x)?, vec![2, 3]);
    Ok(())
}

#[test]
fn flatten_and_reshape_shapes() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (2, 3, 4, 4), &cpu())?;
    assert_eq!(output_dims(Flatten::new(), &x)?, vec![2, 48]);

    let x2 = Tensor::randn(0.0f32, 1.0, (2, 12), &cpu())?;
    assert_eq!(output_dims(Reshape::new([3, 4]), &x2)?, vec![2, 3, 4]);
    Ok(())
}

#[test]
fn normalization_shapes() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (4, 6), &cpu())?;
    assert_eq!(output_dims(BatchNorm::new(6), &x)?, vec![4, 6]);

    let x3 = Tensor::randn(0.0f32, 1.0, (2, 3, 8), &cpu())?;
    assert_eq!(output_dims(LayerNorm::new(8), &x3)?, vec![2, 3, 8]);
    Ok(())
}

#[test]
fn dropout_is_identity_in_eval() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (4, 5), &cpu())?;
    let mut model = Sequential::new().with_device(cpu());
    model.add(Dropout::new(0.5));
    model.compile(Optimizer::sgd(0.01), Loss::MSE, vec![])?;
    // In evaluation mode dropout must pass the input through unchanged.
    let y = model.predict(&x)?;
    assert_eq!(x.to_vec2::<f32>()?, y.to_vec2::<f32>()?);
    Ok(())
}

#[test]
fn embedding_shape() -> Result<()> {
    let idx = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (2, 3), &cpu())?;
    assert_eq!(output_dims(Embedding::new(10, 8), &idx)?, vec![2, 3, 8]);
    Ok(())
}

#[test]
fn recurrent_shapes() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (2, 5, 3), &cpu())?;
    // return_sequences == false -> last step only
    assert_eq!(output_dims(Lstm::new(3, 4), &x)?, vec![2, 4]);
    assert_eq!(output_dims(Gru::new(3, 4), &x)?, vec![2, 4]);
    assert_eq!(output_dims(SimpleRNN::new(3, 4), &x)?, vec![2, 4]);
    // return_sequences == true -> full sequence
    assert_eq!(
        output_dims(Lstm::new(3, 4).with_return_sequences(true), &x)?,
        vec![2, 5, 4]
    );
    Ok(())
}

#[test]
fn attention_shape() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (2, 5, 8), &cpu())?;
    assert_eq!(
        output_dims(MultiHeadAttention::new(8, 2), &x)?,
        vec![2, 5, 8]
    );
    Ok(())
}

/// A heterogeneous stack — Conv -> Pool -> Flatten -> Dense — which is the whole
/// point of the trait-object redesign: a single model holding many layer types.
#[test]
fn mixed_cnn_model() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (4, 1, 8, 8), &cpu())?;

    let mut model = Sequential::new().with_device(cpu());
    model.add(Conv2D::new(1, 8, 3, "relu")); // (4, 8, 6, 6)
    model.add(MaxPool2D::new(2)); //             (4, 8, 3, 3)
    model.add(Flatten::new()); //                (4, 72)
    model.add(Dense::new(10, 8 * 3 * 3, "softmax")); // (4, 10)
    model.compile(Optimizer::adam(0.001), Loss::CrossEntropy, vec![])?;

    let out = model.predict(&x)?;
    assert_eq!(out.dims(), &[4, 10]);

    // Softmax rows sum to 1.
    let sums = out.sum(D::Minus1)?.to_vec1::<f32>()?;
    for s in sums {
        assert!((s - 1.0).abs() < 1e-5, "softmax row did not sum to 1: {s}");
    }
    Ok(())
}

#[test]
fn save_and_load_round_trip() -> Result<()> {
    let x = Tensor::randn(0.0f32, 1.0, (3, 4), &cpu())?;
    let path = std::env::temp_dir().join("kerasty_roundtrip.safetensors");

    let mut a = Sequential::new().with_device(cpu());
    a.add(Dense::new(5, 4, "relu"));
    a.add(Dense::new(2, 5, "linear"));
    a.compile(Optimizer::sgd(0.01), Loss::MSE, vec![])?;
    let p1 = a.predict(&x)?.to_vec2::<f32>()?;
    a.save(&path)?;

    // A freshly initialized model has different weights until we load.
    let mut b = Sequential::new().with_device(cpu());
    b.add(Dense::new(5, 4, "relu"));
    b.add(Dense::new(2, 5, "linear"));
    b.compile(Optimizer::sgd(0.01), Loss::MSE, vec![])?;
    b.load(&path)?;
    let p2 = b.predict(&x)?.to_vec2::<f32>()?;

    assert_eq!(
        p1, p2,
        "loaded model must reproduce the saved model's outputs"
    );

    std::fs::remove_file(&path).ok();
    Ok(())
}
