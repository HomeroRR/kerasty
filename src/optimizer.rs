//! Runtime optimizer state.
//!
//! [`Optimizer`] (in `common::definitions`) is the *description* users pass to
//! `compile`; `OptimizerInstance` is the live Candle optimizer bound to the
//! model's variables. It is an implementation detail and intentionally
//! `pub(crate)` — users never construct or name it.

use candle_core::{Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer as CandleOptimizer, ParamsAdamW, SGD};

use crate::common::definitions::Optimizer;

/// A constructed, ready-to-step optimizer bound to a set of variables.
pub(crate) enum OptimizerInstance {
    Sgd(SGD),
    Adam(AdamW),
}

impl OptimizerInstance {
    /// Build the concrete optimizer for `spec` over `vars`.
    pub(crate) fn build(spec: &Optimizer, vars: Vec<Var>) -> Result<Self> {
        match *spec {
            Optimizer::Sgd { lr } => Ok(OptimizerInstance::Sgd(SGD::new(vars, lr)?)),
            Optimizer::Adam {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                let params = ParamsAdamW {
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                };
                Ok(OptimizerInstance::Adam(AdamW::new(vars, params)?))
            }
        }
    }

    /// Perform one backward + parameter-update step for `loss`.
    pub(crate) fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        match self {
            OptimizerInstance::Sgd(sgd) => sgd.backward_step(loss),
            OptimizerInstance::Adam(adam) => adam.backward_step(loss),
        }
    }
}
