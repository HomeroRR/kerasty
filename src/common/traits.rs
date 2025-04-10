pub use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarMap;

use crate::{common::definitions::Optimizer, Loss, Metric};

/* Trait for Layer */
pub trait Layer: Sized {
    fn init(&self, varmap : &VarMap, dtype: DType, dev: &Device, name: &str) -> Result<Self>;
}

/* Trait for Model */
pub trait Model {
    fn compile(&mut self, optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) -> Result<()>;
    fn fit(&mut self, x: Tensor, y: Tensor, epochs: u64) -> Result<()>;
    fn predict(&self, x: &Tensor) -> Tensor;
    fn evaluate(&self, x: &Tensor, y: &Tensor) -> Vec<f64>;
}

