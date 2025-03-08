//! Keras for Rust with support for Web Assembly.
//!
//! ## Features
//!
//! - Candle backend
//!
//! # Roadmap of Supported Layers

//! |    Layer   | State |                        Example                            |
//! |------------|-------|-----------------------------------------------------------|
//! |    Dense   |&#9989;| [add](https://docs.rs/kerasty/latest/kerasty/fn.add.html) |

pub use candle_core::{bail, DType, Device, Result, Tensor};

pub mod common;
pub use crate::common::definitions::{
    Activation, Initializer, Loss, Metric, Optimizer, Regularizer,
};
pub use crate::common::traits::{Layer, Model};

pub mod layer;
pub use crate::layer::dense::Dense;

pub mod sequential;
pub use crate::sequential::Sequential;
