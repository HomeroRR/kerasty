use candle_core::{DType, Device, Result, Tensor};
use candle_nn::loss::{binary_cross_entropy_with_logit, cross_entropy, mse, nll};
use candle_nn::Optimizer as CandleOptimizer;
use candle_nn::{AdamW, Module, ParamsAdamW, VarMap, SGD};

use crate::common::definitions::{Loss, Metric, Optimizer, OptimizerInstance};
use crate::common::traits::{Layer, Model};

/* Define the Sequential model */
pub struct Sequential<T>
where
    T: Module + 'static + Layer,
{
    layers: Vec<T>,
    optimizer: Optimizer,
    loss: Loss,
    metrics: Vec<Metric>,
    seq: candle_nn::Sequential,
    varmap: VarMap,
    dtype: DType,
    dev: candle_core::Device,
}

impl<T> Sequential<T>
where
    T: Module + 'static + Layer,
{
    /* Create a new Sequential model */
    pub fn new() -> Self {
        let optimizer = Optimizer::SGD(0.01);
        let loss = Loss::MSE;
        let metrics = vec![Metric::Accuracy];
        let seq = candle_nn::seq();
        let dtype = DType::F64;
        let dev = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
        let varmap = VarMap::new();

        Sequential {
            layers: Vec::new(),
            optimizer: optimizer,
            loss: loss,
            metrics: metrics,
            seq: seq,
            varmap: varmap,
            dtype: dtype,
            dev: dev,
        }
    }

    pub fn add(&mut self, layer: T) {
        self.layers.push(layer);
    }
}

impl<T> Model for Sequential<T>
where
    T: Module + 'static + Layer,
{
    fn compile(&mut self, optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) -> Result<()> {
        self.optimizer = optimizer;
        self.loss = loss;
        self.metrics = metrics;

        // Add layers to the model
        let mut seq = candle_nn::seq();
        for (i, layer) in self.layers.iter().enumerate() {
            let layer = layer.init(&self.varmap, self.dtype, &self.dev, &i.to_string())?;
            seq = seq.add(layer);
        }
        self.seq = seq;

        Ok(())
    }

    fn fit(&mut self, x: Tensor, y: Tensor, epochs: u64) -> Result<()> {
        let loss = match self.loss {
            Loss::MSE => mse,
            Loss::NLL => nll,
            Loss::BinaryCrossEntropyWithLogit => binary_cross_entropy_with_logit,
            Loss::CrossEntropy => cross_entropy,
        };
        let mut optimizer: OptimizerInstance = match self.optimizer {
            Optimizer::SGD(lr) => OptimizerInstance::SGD(SGD::new(self.varmap.all_vars(), lr)?),
            Optimizer::Adam(lr, beta1, beta2, eps, weight_decay) => {
                let params = ParamsAdamW {
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                };
                OptimizerInstance::Adam(AdamW::new(self.varmap.all_vars(), params)?)
            }
        };

        /* Train the model*/
        for _e in 0..epochs {
            // Forward pass
            let y_pred = self.seq.forward(&x)?;
            // Compute loss
            let loss_value = loss(&y_pred, &y)?;
            // Update weights
            match optimizer {
                OptimizerInstance::SGD(ref mut sgd) => {
                    sgd.backward_step(&loss_value)?;
                }
                OptimizerInstance::Adam(ref mut adam) => {
                    adam.backward_step(&loss_value)?;
                }
            };
        }
        Ok(())
    }

    fn predict(&self, x: &Tensor) -> Tensor {
        self.seq.forward(x).unwrap()
    }

    fn evaluate(&self, x: &Tensor, y: &Tensor) -> f64 {
        let y_pred = self.predict(&x);
        let loss = match self.loss {
            Loss::MSE => mse,
            Loss::NLL => nll,
            Loss::BinaryCrossEntropyWithLogit => binary_cross_entropy_with_logit,
            Loss::CrossEntropy => cross_entropy,
        };
        let sum_loss = loss(&y_pred, &y).unwrap().to_vec0::<f64>().unwrap();
        let avg_loss = sum_loss / y.dims2().unwrap().0 as f64;
        avg_loss
    }
}
