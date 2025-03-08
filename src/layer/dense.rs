pub use candle_core::{bail, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::common::definitions::{
    Activation, Initializer, Regularizer,
};
use crate::common::traits::Layer;

/* Function for creatin a Dense layer from Candle in the style of Keras */
#[derive(Clone, Debug)]
pub struct Dense {
    units: usize,
    input_dim: usize,
    activation: Activation,
    kernel_initializer: Option<Initializer>,
    bias_initializer: Option<Initializer>,
    kernel_regularizer: Option<Regularizer>,
    bias_regularizer: Option<Regularizer>,
    linear: Option<Linear>,
}

impl Dense {
    pub fn new(
        units: usize,
        input_dim: usize,
        activation: Activation,
        kernel_initializer: Option<Initializer>,
        bias_initializer: Option<Initializer>,
        kernel_regularizer: Option<Regularizer>,
        bias_regularizer: Option<Regularizer>,
    ) -> Self {
        Dense {
            units,
            input_dim,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            linear: None,
        }
    }
}

impl Layer for Dense {
    fn init(&self, vb: VarBuilder) -> Result<Self> {
        let (out_dim, in_dim) = (self.units, self.input_dim);
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
        let bs = vb.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
        let linear = Linear::new(ws, Some(bs));
        Ok(Self {
            linear: Some(linear),
            ..self.clone()
        })
    }
}

impl Module for Dense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let Some(linear) = &self.linear {
            linear.forward(xs)
        } else {
            bail!("Linear module not initialized")
        }
    }
}
