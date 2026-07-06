//! Shape-manipulation layers: `Flatten` and `Reshape`. No trainable parameters.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::common::traits::Layer;

/// Flattens all non-batch dimensions into one, e.g. `(batch, c, h, w) ->
/// (batch, c*h*w)`. Typically placed between convolutional and dense layers.
#[derive(Clone, Debug, Default)]
pub struct Flatten;

impl Flatten {
    /// Create a flatten layer.
    pub fn new() -> Self {
        Flatten
    }
}

impl Layer for Flatten {
    fn build(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        xs.flatten_from(1)
    }

    fn kind(&self) -> &'static str {
        "Flatten"
    }
}

/// Reshapes each sample to a fixed target shape while preserving the batch
/// dimension. The provided `shape` describes a single sample (no batch axis).
#[derive(Clone, Debug)]
pub struct Reshape {
    shape: Vec<usize>,
}

impl Reshape {
    /// Create a reshape layer producing samples of the given per-sample `shape`.
    pub fn new(shape: impl Into<Vec<usize>>) -> Self {
        Reshape {
            shape: shape.into(),
        }
    }
}

impl Layer for Reshape {
    fn build(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let batch = xs.dim(0)?;
        let mut dims = Vec::with_capacity(self.shape.len() + 1);
        dims.push(batch);
        dims.extend_from_slice(&self.shape);
        xs.reshape(dims)
    }

    fn kind(&self) -> &'static str {
        "Reshape"
    }
}
