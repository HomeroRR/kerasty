//! Fully-connected (dense) layer.

use candle_core::{bail, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::common::definitions::{Activation, Initializer};
use crate::common::traits::Layer;

/// A densely-connected layer: `activation(x · Wᵀ + b)`.
///
/// ```
/// use kerasty::Dense;
/// // 64 output units, 128 input features, ReLU activation
/// let layer = Dense::new(64, 128, "relu");
/// ```
#[derive(Clone, Debug)]
pub struct Dense {
    units: usize,
    input_dim: usize,
    activation: Activation,
    initializer: Initializer,
    linear: Option<Linear>,
}

impl Dense {
    /// Create a dense layer with `units` outputs, `input_dim` inputs and the
    /// named `activation` (see [`Activation::from_str`]). An unknown activation
    /// name falls back to linear.
    pub fn new(units: usize, input_dim: usize, activation: &str) -> Self {
        Dense {
            units,
            input_dim,
            activation: activation.parse().unwrap_or_default(),
            initializer: Initializer::default(),
            linear: None,
        }
    }

    /// Builder: set the kernel initializer.
    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    /// Builder: set the activation from an [`Activation`] value.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl Layer for Dense {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        // `candle_nn::linear` seeds sensible defaults; the initializer is kept
        // for API parity and future per-layer control.
        let _ = self.initializer;
        self.linear = Some(linear(self.input_dim, self.units, vb)?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let Some(linear) = &self.linear else {
            bail!("Dense layer used before `build`/`compile`");
        };
        let z = linear.forward(xs)?;
        self.activation.apply(&z)
    }

    fn kind(&self) -> &'static str {
        "Dense"
    }
}
