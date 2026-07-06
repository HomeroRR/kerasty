//! Dropout regularization layer.

use candle_core::{Result, Tensor};
use candle_nn::{Dropout as CandleDropout, VarBuilder};

use crate::common::traits::Layer;

/// Randomly zeroes a fraction `rate` of inputs during training and scales the
/// rest; during inference it is a no-op (identity).
#[derive(Clone, Debug)]
pub struct Dropout {
    rate: f32,
    inner: CandleDropout,
}

impl Dropout {
    /// Create a dropout layer that drops inputs with probability `rate`
    /// (in `[0, 1)`).
    pub fn new(rate: f32) -> Self {
        Dropout {
            rate,
            inner: CandleDropout::new(rate),
        }
    }

    /// The configured drop probability.
    pub fn rate(&self) -> f32 {
        self.rate
    }
}

impl Layer for Dropout {
    fn build(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.inner.forward(xs, train)
    }

    fn kind(&self) -> &'static str {
        "Dropout"
    }
}
