//! Keras for Rust with support for Web Assembly.
//!
//! ## Features
//!
//! - Candle backend
//!
//! # Roadmap of Supported Layers

//! |    Layer   | State |                        Example                            |
//! |------------|-------|-----------------------------------------------------------|
//! |    Dense   |&#9989;| [add](https://docs.rs/kerasty/latest/kerasty/fn.add.html) |

/// Adds two numbers.
///
/// # Examples
///
/// ```
/// let result = kerasty::add(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub use candle_core::{bail, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};

/* Define the Activation types */
#[derive(Clone, Debug)]
pub enum Activation {
    Sigmoid,
    ReLU,
    Softmax,
}

/* Define the Initializer types */
#[derive(Clone, Debug)]
pub enum Initializer {
    Zeros,
    Ones,
    RandomNormal,
    RandomUniform,
}

/* Define the Regularizer types */
#[derive(Clone, Debug)]
pub enum Regularizer {
    L1,
    L2,
}

/* Define the Optimizer types */
#[derive(Clone, Debug)]
pub enum Optimizer {
    SGD(f64),
    Adam(f64, f64, f64),
}

/* Define the Loss types */
#[derive(Clone, Debug)]
pub enum Loss {
    MeanSquaredError,
    CrossEntropy,
}

/* Define the Metric types */
#[derive(Clone, Debug)]
pub enum Metric {
    MeanSquaredError,
    Accuracy,
}

pub trait Layer: Sized {
    fn init(&self, vb: VarBuilder) -> Result<Self>;
}

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

/* Trait for Model */
pub trait Model {
    fn compile(&mut self, optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) -> Result<()>;
    fn fit(&mut self, x: Tensor, y: Tensor, epochs: u64, batch_size: u64) -> Result<()>;
    fn predict(&self, x: &Tensor) -> Tensor;
    fn evaluate(&self, x: Tensor, y: Tensor) -> f64;
}

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
