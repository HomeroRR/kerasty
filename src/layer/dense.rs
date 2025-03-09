pub use candle_core::{bail, DType, Device, Result, Tensor};
use candle_nn::ops::{sigmoid, softmax};
use candle_nn::{Linear, Module, VarBuilder, VarMap};

use crate::common::definitions::Activation;
use crate::common::traits::Layer;

/* Function for creatin a Dense layer from Candle in the style of Keras */
#[derive(Clone, Debug)]
pub struct Dense {
    units: usize,
    input_dim: usize,
    activation: Activation,
    linear: Option<Linear>,
}

impl Dense {
    pub fn new(units: usize, input_dim: usize, activation: &str) -> Self {
        let activation = activation.parse().unwrap_or(Activation::ReLU);
        Dense {
            units,
            input_dim,
            activation: activation,
            linear: None,
        }
    }
}

impl Layer for Dense {
    fn init(&self, varmap: &VarMap, dtype: DType, dev: &Device, name: &str) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, dtype, &dev).pp(name);
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
            let res = linear.forward(xs)?;
            match self.activation {
                Activation::Linear => Ok(res),
                Activation::Sigmoid => Ok(sigmoid(&res)?),
                Activation::ReLU => Ok(res.relu()?),
                Activation::Softmax => Ok(softmax(&res, 1)?),
            }
        } else {
            bail!("Linear module not initialized")
        }
    }
}
