//! Normalization layers: `BatchNorm` and `LayerNorm`.

use candle_core::{bail, Result, Tensor};
use candle_nn::{
    batch_norm, layer_norm, BatchNorm as CandleBatchNorm, BatchNormConfig,
    LayerNorm as CandleLayerNorm, LayerNormConfig, Module, ModuleT, VarBuilder,
};

use crate::common::traits::Layer;

/// Batch normalization over the channel/feature dimension.
///
/// In training it normalizes using the current batch statistics and updates the
/// running mean/variance; in inference it uses the accumulated running stats.
#[derive(Clone, Debug)]
pub struct BatchNorm {
    num_features: usize,
    eps: f64,
    inner: Option<CandleBatchNorm>,
}

impl BatchNorm {
    /// Create a batch-norm layer for inputs with `num_features` channels.
    pub fn new(num_features: usize) -> Self {
        BatchNorm {
            num_features,
            eps: 1e-5,
            inner: None,
        }
    }

    /// Builder: override the numerical-stability epsilon.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl Layer for BatchNorm {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        let cfg = BatchNormConfig {
            eps: self.eps,
            ..Default::default()
        };
        self.inner = Some(batch_norm(self.num_features, cfg, vb)?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let Some(bn) = &self.inner else {
            bail!("BatchNorm layer used before `build`/`compile`");
        };
        bn.forward_t(xs, train)
    }

    fn kind(&self) -> &'static str {
        "BatchNorm"
    }
}

/// Layer normalization over the last dimension (`normalized_shape == size`).
#[derive(Clone, Debug)]
pub struct LayerNorm {
    size: usize,
    eps: f64,
    inner: Option<CandleLayerNorm>,
}

impl LayerNorm {
    /// Create a layer-norm over a feature dimension of `size`.
    pub fn new(size: usize) -> Self {
        LayerNorm {
            size,
            eps: 1e-5,
            inner: None,
        }
    }

    /// Builder: override the numerical-stability epsilon.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl Layer for LayerNorm {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        let cfg = LayerNormConfig {
            eps: self.eps,
            ..Default::default()
        };
        self.inner = Some(layer_norm(self.size, cfg, vb)?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let Some(ln) = &self.inner else {
            bail!("LayerNorm layer used before `build`/`compile`");
        };
        ln.forward(xs)
    }

    fn kind(&self) -> &'static str {
        "LayerNorm"
    }
}
