# kerasty

**Keras for Rust**, powered by [Candle](https://github.com/huggingface/candle), with WebAssembly support.

[![Crates.io][crates-badge]][crates-url]
[![MIT licensed][mit-badge]][mit-url]
[![Discord chat][discord-badge]][discord-url]

[crates-badge]: https://img.shields.io/crates/v/kerasty.svg
[crates-url]: https://crates.io/crates/kerasty
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://github.com/HomeroRR/kerasty/blob/main/LICENSE
[discord-badge]: https://img.shields.io/badge/Discord--0
[discord-url]: https://discord.gg/cZQnB8zX2H

Kerasty gives you a familiar, Keras-style API for building, training, and running
neural networks in Rust. Stack any mix of layers into a `Sequential` model,
`compile` it with an optimizer / loss / metrics, then `fit`, `predict`, and
`evaluate`. Trained models save to [safetensors](https://github.com/huggingface/safetensors)
and reload for inference — including in the browser via `wasm32`.

Built on **Candle 0.11**.

## Roadmap of Supported Layers

| Layer | State | Notes |
|-------|:-----:|-------|
| Dense | ✅ | [`Dense`](https://docs.rs/kerasty/latest/kerasty/layer/dense/struct.Dense.html) |
| Convolution (1D / 2D) | ✅ | `Conv1D`, `Conv2D` with stride, dilation, `valid`/`same` padding |
| Pooling | ✅ | `MaxPool2D`, `AvgPool2D`, `GlobalAveragePooling2D` |
| Flatten / Reshape | ✅ | `Flatten`, `Reshape` |
| Normalization | ✅ | `BatchNorm`, `LayerNorm` |
| Dropout | ✅ | `Dropout` (train-only) |
| Embedding | ✅ | `Embedding` |
| Recurrent | ✅ | `SimpleRNN`, `Lstm`, `Gru` (with `return_sequences`) |
| Attention | ✅ | `MultiHeadAttention` |
| BERT / Llama | 🏗️ | Composable today from `Embedding` + `MultiHeadAttention` + `LayerNorm` + `Dense`; prebuilt model zoo is on the roadmap |

Other building blocks: rich [`Activation`]s (relu, gelu, silu/swish, tanh, elu,
leaky_relu, mish, softmax, …), [`Optimizer`]s (`sgd`, `adam`/AdamW),
[`Loss`]es (MSE, MAE, Huber, cross-entropy, NLL, BCE), [`Metric`]s
(accuracy, categorical accuracy, MSE, MAE), [`Initializer`]s and [`Padding`].

[`Activation`]: https://docs.rs/kerasty/latest/kerasty/enum.Activation.html
[`Optimizer`]: https://docs.rs/kerasty/latest/kerasty/enum.Optimizer.html
[`Loss`]: https://docs.rs/kerasty/latest/kerasty/enum.Loss.html
[`Metric`]: https://docs.rs/kerasty/latest/kerasty/enum.Metric.html
[`Initializer`]: https://docs.rs/kerasty/latest/kerasty/enum.Initializer.html
[`Padding`]: https://docs.rs/kerasty/latest/kerasty/enum.Padding.html

## Installation

```toml
[dependencies]
kerasty = "0.3"
```

## Example — the classic XOR problem

Solution to the classic [XOR problem](https://www.geeksforgeeks.org/how-neural-networks-solve-the-xor-problem):

```rust,no_run
use kerasty::{Dense, Device, Loss, Metric, Model, Optimizer, Result, Sequential, Tensor};

fn main() -> Result<()> {
    // XOR inputs and targets (f32 — the model's default dtype).
    let x_data: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x = Tensor::from_slice(&x_data, (4, 2), &Device::Cpu)?;
    let y_data: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];
    let y = Tensor::from_slice(&y_data, (4, 1), &Device::Cpu)?;

    // Build the network.
    let mut model = Sequential::new();
    model.add(Dense::new(8, 2, "tanh"));
    model.add(Dense::new(1, 8, "sigmoid"));

    // Compile: sigmoid output pairs with BinaryCrossEntropy (probabilities in).
    model.compile(
        Optimizer::adam(0.05),
        Loss::BinaryCrossEntropy,
        vec![Metric::Accuracy],
    )?;

    // Train.
    model.fit(x.clone(), y.clone(), 3_000)?;

    // Predict.
    let predictions = model.predict(&x)?.reshape(4)?.to_vec1::<f32>()?;
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

    // Evaluate.
    let score = model.evaluate(&x, &y)?;
    println!("Average loss: {}", score[0]);
    println!("Accuracy: {}", score[1]);
    Ok(())
}
```

Expected output:

```text
Predictions:
Input: [0.0, 0.0] => Predicted Output: 0, Actual Output: 0
Input: [0.0, 1.0] => Predicted Output: 1, Actual Output: 1
Input: [1.0, 0.0] => Predicted Output: 1, Actual Output: 1
Input: [1.0, 1.0] => Predicted Output: 0, Actual Output: 0
```

## Example — a convolutional network (mixed layers)

The whole point of `Sequential` is that it holds a **heterogeneous** stack of
layers — convolutions, pooling, and dense layers in the same model:

```rust,no_run
use kerasty::{Conv2D, Dense, Flatten, MaxPool2D, Model, Optimizer, Loss, Metric, Sequential};

let mut model = Sequential::new();
model.add(Conv2D::new(1, 16, 3, "relu"));  // (N, 1, 28, 28) -> (N, 16, 26, 26)
model.add(MaxPool2D::new(2));              //                -> (N, 16, 13, 13)
model.add(Flatten::new());                 //                -> (N, 16*13*13)
model.add(Dense::new(10, 16 * 13 * 13, "softmax"));

model.compile(
    Optimizer::adam(0.001),
    Loss::CrossEntropy,
    vec![Metric::CategoricalAccuracy],
)?;

model.summary();
// then: model.fit(x, y, epochs)?  /  model.predict(&x)?
```

## Saving and loading

```rust,no_run
// Persist trainable parameters as safetensors...
model.save("model.safetensors")?;

// ...and reload them into a model with the same architecture.
let mut restored = Sequential::new();
restored.add(Dense::new(2, 2, "relu"));
restored.compile(Optimizer::adam(0.001), Loss::MSE, vec![])?;
restored.load("model.safetensors")?;
```

## WebAssembly (inference)

Kerasty compiles to `wasm32-unknown-unknown` for in-browser inference. Training
(the optimizer step) targets native. `f32` is the default dtype and is
recommended for wasm.

```sh
./dev.sh wasm
```

This installs the `wasm32-unknown-unknown` target if needed and builds the crate.
WebAssembly RNG (used by weight initialization) is enabled through the `wasm_js`
feature of `getrandom` plus a cfg flag in [`.cargo/config.toml`](.cargo/config.toml);
both are already configured.

## Development

A helper script wraps the common tasks:

```sh
./dev.sh fmt        # format
./dev.sh lint       # clippy, warnings-as-errors
./dev.sh test       # run the test suite
./dev.sh build      # native build
./dev.sh wasm       # wasm32 inference build
./dev.sh doc        # build & open API docs
./dev.sh check      # fmt-check + lint + test (CI gate)
./dev.sh all        # everything above
```

## License

MIT

Copyright © 2025-2035 Homero Roman Roman
Copyright © 2025-2035 Frederick Roman

## Contributing

Contributions are welcome.
Please open an issue or a pull request to report a bug or request a feature.
