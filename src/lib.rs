//! # Kerasty — Keras for Rust, powered by [Candle], with WebAssembly support.
//!
//! Kerasty gives you a familiar, Keras-style API for building, training and
//! running neural networks in Rust. Build a [`Sequential`] stack from any mix of
//! [layers](crate::layer), `compile` it with an [`Optimizer`], [`Loss`] and
//! [`Metric`]s, then `fit`, `predict` and `evaluate`. Trained models can be
//! saved as safetensors and loaded for inference — including in the browser via
//! `wasm32`.
//!
//! ## Roadmap of Supported Layers
//!
//! | Layer | State |
//! |-------|-------|
//! | Dense | ✅ |
//! | Convolution (1D/2D) | ✅ |
//! | Pooling (Max/Avg/GlobalAvg) | ✅ |
//! | Flatten / Reshape | ✅ |
//! | Normalization (Batch/Layer) | ✅ |
//! | Dropout | ✅ |
//! | Embedding | ✅ |
//! | Recurrent (SimpleRNN/LSTM/GRU) | ✅ |
//! | Attention (Multi-Head) | ✅ |
//! | BERT / Llama | 🏗️ composable via [`MultiHeadAttention`] + [`Embedding`] + [`LayerNorm`] |
//!
//! ## Example — the classic XOR problem
//!
//! ```no_run
//! use kerasty::{Dense, Device, Loss, Metric, Model, Optimizer, Result, Sequential, Tensor};
//!
//! fn xor() -> Result<()> {
//!     let x = Tensor::from_slice(&[0f32, 0., 0., 1., 1., 0., 1., 1.], (4, 2), &Device::Cpu)?;
//!     let y = Tensor::from_slice(&[0f32, 1., 1., 0.], (4, 1), &Device::Cpu)?;
//!
//!     let mut model = Sequential::new();
//!     model.add(Dense::new(8, 2, "tanh"));
//!     model.add(Dense::new(1, 8, "sigmoid"));
//!
//!     // sigmoid output pairs with BinaryCrossEntropy (probabilities in).
//!     model.compile(
//!         Optimizer::adam(0.05),
//!         Loss::BinaryCrossEntropy,
//!         vec![Metric::Accuracy],
//!     )?;
//!     model.fit(x.clone(), y.clone(), 3_000)?;
//!
//!     let score = model.evaluate(&x, &y)?;
//!     println!("loss = {:.4}, accuracy = {:.2}", score[0], score[1]);
//!     Ok(())
//! }
//! ```
//!
//! [Candle]: https://github.com/huggingface/candle
//!
//! # License
//! MIT — Copyright © 2025-2035 Homero Roman Roman, Frederick Roman.

#![warn(missing_docs)]

// Re-export the handful of Candle types users interact with directly, so a
// single `use kerasty::*;` is enough to get started.
pub use candle_core::{bail, DType, Device, Result, Tensor, D};

pub mod common;
pub mod layer;
pub mod sequential;

mod optimizer;

pub use crate::common::definitions::{
    Activation, Initializer, Loss, Metric, Optimizer, Padding, Regularizer,
};
pub use crate::common::traits::{Layer, Model};

pub use crate::layer::{
    AvgPool2D, BatchNorm, Conv1D, Conv2D, Dense, Dropout, Embedding, Flatten,
    GlobalAveragePooling2D, Gru, LayerNorm, Lstm, MaxPool2D, MultiHeadAttention, Reshape,
    SimpleRNN,
};
pub use crate::sequential::Sequential;

/// Convenience prelude: `use kerasty::prelude::*;` pulls in everything needed to
/// build, train and run a model.
pub mod prelude {
    pub use crate::common::definitions::{
        Activation, Initializer, Loss, Metric, Optimizer, Padding, Regularizer,
    };
    pub use crate::common::traits::{Layer, Model};
    pub use crate::layer::{
        AvgPool2D, BatchNorm, Conv1D, Conv2D, Dense, Dropout, Embedding, Flatten,
        GlobalAveragePooling2D, Gru, LayerNorm, Lstm, MaxPool2D, MultiHeadAttention, Reshape,
        SimpleRNN,
    };
    pub use crate::sequential::Sequential;
    pub use candle_core::{DType, Device, Result, Tensor, D};
}
