# kerasty
Keras for Candle (Rust ML framework) with support for Web Assembly.

[![Crates.io][crates-badge]][crates-url]
[![MIT licensed][mit-badge]][mit-url]
[![Discord chat][discord-badge]][discord-url]

[crates-badge]:  https://img.shields.io/badge/kerasty-0
[crates-url]: https://crates.io/crates/kerasty
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://github.com/HomeroRR/kerasty/blob/main/LICENSE
[discord-badge]: https://img.shields.io/badge/Discord--0
[discord-url]: https://discord.gg/cZQnB8zX2H



# Roadmap of Supported Layers

 |       Layer      | State |                        Example                            |
 |------------------|---|-----------------------------------------------------------|
 |    Dense         |✅| [Dense](https://docs.rs/kerasty/latest/kerasty/layer/dense/struct.Dense.html) |
 |    Convolution   |🏗️| CNN|
 |    Normalization |🏗️| Norm|
 |    Flatten       |🏗️| Flatten|
 |    Pooling       |🏗️| Pool|
 |    Recurrent     |🏗️| RNN|
 |    Attention     |🏗️| Attn|
 |    Bert          |🏗️| BERT|
 |    Llama         |🏗️| LLAMA|

# Examples
Solution to the classic [XOR problem](https://www.geeksforgeeks.org/how-neural-networks-solve-the-xor-problem)

```rust,no_run
use kerasty::{Dense, Device, Loss, Metric, Model, Optimizer, Sequential, Tensor};

// Define the XOR input and output data
let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
let x = Tensor::from_slice(&x_data, (4, 2), &Device::Cpu)?;
let y_data = vec![0.0, 1.0, 1.0, 0.0];
let y = Tensor::from_slice(&y_data, (4, 1), &Device::Cpu)?;

// Build the neural network model
let mut model = Sequential::new();
model.add(Dense::new(2, 2, "relu"));
model.add(Dense::new(1, 2, "sigmoid"));

// Compile the model
model.compile(
    Optimizer::Adam(0.001, 0.9, 0.999, 1e-8, 0.0),
    Loss::BinaryCrossEntropyWithLogit,
    vec![Metric::Accuracy],
)?;

// Train the model
model.fit(x.clone(), y.clone(), 10000)?;

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
```
The expected output is as follows:
```shell,no_run
Predictions:
Input: [0.0, 0.0] => Predicted Output: 0, Actual Output: 0
Input: [0.0, 1.0] => Predicted Output: 1, Actual Output: 1
Input: [1.0, 0.0] => Predicted Output: 1, Actual Output: 1
Input: [1.0, 1.0] => Predicted Output: 0, Actual Output: 0
```
# License
MIT

Copyright © 2025-2035 Homero Roman Roman  
Copyright © 2025-2035 Frederick Roman

# Contributing

Contributions are welcome.  
Please open an issue or a pull request to report a bug or request a feature.
