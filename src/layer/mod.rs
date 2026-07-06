//! Built-in layers. Every layer implements [`Layer`](crate::Layer) and can be
//! freely mixed inside a [`Sequential`](crate::Sequential) model.

pub mod attention;
pub mod conv;
pub mod dense;
pub mod dropout;
pub mod embedding;
pub mod flatten;
pub mod normalization;
pub mod pooling;
pub mod recurrent;

pub use attention::MultiHeadAttention;
pub use conv::{Conv1D, Conv2D};
pub use dense::Dense;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::{Flatten, Reshape};
pub use normalization::{BatchNorm, LayerNorm};
pub use pooling::{AvgPool2D, GlobalAveragePooling2D, MaxPool2D};
pub use recurrent::{Gru, Lstm, SimpleRNN};
