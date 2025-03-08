use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};

use crate::common::definitions::{Loss, Metric, Optimizer};
use crate::common::traits::{Layer, Model};

/* Define the Sequential model */
pub struct Sequential<'a, T: Module + 'static + Layer> {
    layers: Vec<T>,
    optimizer: Option<Optimizer>,
    loss: Option<Loss>,
    metrics: Option<Vec<Metric>>,
    seq: Option<candle_nn::Sequential>,
    vb: VarBuilder<'a>,
    dev: candle_core::Device,
    dtype: DType,
}

impl<'a, T> Sequential<'a, T>
where
    T: Module + 'static + Layer,
{
    /* Create a new Sequential model */
    pub fn new() -> Self {
        let seq = candle_nn::seq();
        let varmap = VarMap::new();
        let dev = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
        let dtype = DType::F32;
        let vb = VarBuilder::from_varmap(&varmap, dtype, &dev);
        Sequential {
            layers: Vec::new(),
            optimizer: None,
            loss: None,
            metrics: None,
            seq: Some(seq),
            vb: vb,
            dev: Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu),
            dtype: DType::F32,
        }
    }

    pub fn add(&mut self, layer: T) {
        self.layers.push(layer);
    }
}

impl<'a, T> Model for Sequential<'a, T>
where
    T: Module + 'static + Layer,
{
    fn compile(&mut self, optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) -> Result<()> {
        self.optimizer = Some(optimizer);
        self.loss = Some(loss);
        self.metrics = Some(metrics);

        // Add layers to the model
        let mut seq = candle_nn::seq();

        for layer in &self.layers {
            let layer = layer.init(self.vb.clone())?;
            seq = seq.add(layer);
        }
        self.seq = Some(seq);
        Ok(())
    }

    fn fit(&mut self, x: Tensor, y: Tensor, epochs: u64, batch_size: u64) -> Result<()> {
        Ok(())
    }

    fn predict(&self, x: &Tensor) -> Tensor {
        let first_column = x.narrow(1, 0, 1).unwrap();
        first_column
    }

    fn evaluate(&self, x: Tensor, y: Tensor) -> f64 {
        0.0
    }
}
