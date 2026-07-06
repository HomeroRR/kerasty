//! The `Sequential` model: a linear stack of heterogeneous layers.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::common::definitions::{Loss, Metric, Optimizer};
use crate::common::traits::{Layer, Model};
use crate::optimizer::OptimizerInstance;

/// A Keras-style linear stack of layers.
///
/// Unlike a design generic over a single layer type, `Sequential` stores
/// `Box<dyn Layer>`, so any mix of layer types can be stacked:
///
/// ```no_run
/// use kerasty::{Sequential, Conv2D, MaxPool2D, Flatten, Dense};
/// let mut model = Sequential::new();
/// model.add(Conv2D::new(1, 16, 3, "relu"));
/// model.add(MaxPool2D::new(2));
/// model.add(Flatten::new());
/// model.add(Dense::new(10, 16 * 13 * 13, "softmax"));
/// ```
///
/// All parameters live in one shared [`VarMap`], which the optimizer trains and
/// [`save`](Sequential::save) / [`load`](Sequential::load) persist as
/// safetensors.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    varmap: VarMap,
    device: Device,
    dtype: DType,
    optimizer: Optimizer,
    loss: Loss,
    metrics: Vec<Metric>,
    built: bool,
    verbose: bool,
}

impl Sequential {
    /// Create an empty model. Defaults: CUDA if available else CPU, `f32` dtype
    /// (as in Keras, and required by some fused kernels such as LayerNorm),
    /// Adam(1e-3) optimizer, MSE loss.
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
            varmap: VarMap::new(),
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            dtype: DType::F32,
            optimizer: Optimizer::default(),
            loss: Loss::MSE,
            metrics: vec![Metric::Accuracy],
            built: false,
            verbose: false,
        }
    }

    /// Builder: choose the compute device (CPU, CUDA, …).
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Builder: choose the parameter/compute dtype (`f32` default, which suits
    /// WebAssembly inference and fused kernels; use `f64` for extra precision on
    /// layers that support it).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Builder: print the loss periodically during [`fit`](Model::fit).
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Append a layer to the stack.
    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// The device the model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run the stack, threading the `train` flag through every layer.
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out, train)?;
        }
        Ok(out)
    }

    /// Print a Keras-style summary of the layer stack.
    pub fn summary(&self) {
        println!("Model: Sequential");
        println!("{:-<48}", "");
        println!("{:<4} {:<28} {:<14}", "#", "Layer", "Trainable");
        println!("{:-<48}", "");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("{:<4} {:<28} {:<14}", i, layer.kind(), "yes");
        }
        println!("{:-<48}", "");
        let n_params: usize = self.varmap.all_vars().iter().map(|v| v.elem_count()).sum();
        println!("Total trainable parameters: {n_params}");
        println!("{:-<48}", "");
    }

    /// Save all trainable parameters to a safetensors file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path)
    }

    /// Load trainable parameters from a safetensors file. The model must already
    /// be [`compile`](Model::compile)d (so the variables exist) before loading.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.varmap.load(path)
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for Sequential {
    fn compile(&mut self, optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) -> Result<()> {
        self.optimizer = optimizer;
        self.loss = loss;
        self.metrics = metrics;

        let root = VarBuilder::from_varmap(&self.varmap, self.dtype, &self.device);
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let name = format!("layer_{i}_{}", layer.kind().to_ascii_lowercase());
            layer.build(root.pp(&name))?;
        }
        self.built = true;
        Ok(())
    }

    fn fit(&mut self, x: Tensor, y: Tensor, epochs: u64) -> Result<()> {
        if !self.built {
            candle_core::bail!("call `compile` before `fit`");
        }
        let mut optimizer = OptimizerInstance::build(&self.optimizer, self.varmap.all_vars())?;
        for epoch in 0..epochs {
            let y_pred = self.forward(&x, true)?;
            let loss = self.loss.compute(&y_pred, &y)?;
            optimizer.backward_step(&loss)?;
            if self.verbose && (epoch + 1) % 100 == 0 {
                let value = loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
                println!("epoch {:>5} - loss {value:.6}", epoch + 1);
            }
        }
        Ok(())
    }

    fn predict(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x, false)
    }

    fn evaluate(&self, x: &Tensor, y: &Tensor) -> Result<Vec<f64>> {
        let y_pred = self.predict(x)?;
        let loss = self
            .loss
            .compute(&y_pred, y)?
            .to_dtype(DType::F64)?
            .to_scalar::<f64>()?;
        let mut results = Vec::with_capacity(1 + self.metrics.len());
        results.push(loss);
        for metric in &self.metrics {
            results.push(metric.compute(&y_pred, y)?);
        }
        Ok(results)
    }
}
