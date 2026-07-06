//! The two traits every model is built from: [`Layer`] and [`Model`].

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::common::definitions::{Loss, Metric, Optimizer};

/// A single, composable transformation in a network.
///
/// The trait is deliberately **object-safe** (no `Self` return type, no generic
/// methods) so that a model can store `Box<dyn Layer>` and freely mix layer
/// types — a `Dense`, a `Conv2D` and a `Dropout` in the same stack. This is the
/// key difference from a design generic over a single concrete layer type.
///
/// Implementors typically hold their configuration up front and lazily create
/// their trainable parameters in [`build`](Layer::build), storing the resulting
/// Candle module in an `Option` that [`forward`](Layer::forward) then uses.
pub trait Layer {
    /// Create this layer's trainable parameters into `vb`, which is already
    /// scoped (name-prefixed) to the layer. Stateless layers (e.g. `Flatten`,
    /// pooling) implement this as a no-op.
    fn build(&mut self, vb: VarBuilder) -> Result<()>;

    /// Run the forward pass.
    ///
    /// `train` toggles training-time behavior: dropout is active and batch-norm
    /// updates its running statistics only when `train == true`.
    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor>;

    /// A short, human-readable type name used in `Sequential::summary()`.
    fn kind(&self) -> &'static str;
}

/// The training / inference interface implemented by [`Sequential`](crate::Sequential).
pub trait Model {
    /// Configure the optimizer, loss and metrics, and build all layer parameters.
    fn compile(&mut self, optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) -> Result<()>;

    /// Fit the model to `(x, y)` for `epochs` full-batch steps.
    fn fit(&mut self, x: Tensor, y: Tensor, epochs: u64) -> Result<()>;

    /// Run inference (evaluation mode) and return the predictions.
    fn predict(&self, x: &Tensor) -> Result<Tensor>;

    /// Evaluate the model, returning the average loss followed by each metric.
    fn evaluate(&self, x: &Tensor, y: &Tensor) -> Result<Vec<f64>>;
}
