//! Core building-block enums shared across the crate.
//!
//! These types describe *what* a model should do — which activation to apply,
//! which optimizer to train with, which loss to minimize — while the concrete
//! Candle machinery that carries them out lives in the layer and optimizer
//! modules. Keeping them here, decoupled from any single layer, is what lets a
//! [`Dense`](crate::Dense) and a [`Conv2D`](crate::Conv2D) share the exact same
//! activation code path.

use std::fmt;
use std::str::FromStr;

use candle_core::{bail, DType, Error, Result, Tensor, D};
use candle_nn::init::{FanInOut, NonLinearity, NormalOrUniform};
use candle_nn::loss::{binary_cross_entropy_with_logit, cross_entropy, huber, mse, nll};
use candle_nn::ops::{log_softmax, sigmoid, silu, softmax};
use candle_nn::Init;

/// Element-wise activation functions, mirroring the names used by Keras.
///
/// Construct them directly (`Activation::ReLU`), from a string
/// (`"relu".parse()?` / [`Activation::from_str`]), and apply them uniformly
/// with [`Activation::apply`]. Every layer that supports an activation routes
/// through `apply`, so there is a single, well-tested code path.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Activation {
    /// Identity: `f(x) = x`.
    #[default]
    Linear,
    /// Rectified linear unit: `max(0, x)`.
    ReLU,
    /// Leaky ReLU with the given negative slope (Keras default `0.3`, here `0.01`).
    LeakyReLU(f64),
    /// Exponential linear unit with slope `alpha` for negative inputs.
    ELU(f64),
    /// Logistic sigmoid: `1 / (1 + e^-x)`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Softmax over the last dimension.
    Softmax,
    /// Log-softmax over the last dimension (numerically stable).
    LogSoftmax,
    /// Gaussian error linear unit (tanh approximation).
    GELU,
    /// Sigmoid-weighted linear unit, a.k.a. Swish: `x * sigmoid(x)`.
    SiLU,
    /// Softplus: `ln(1 + e^x)`.
    Softplus,
    /// Mish: `x * tanh(softplus(x))`.
    Mish,
    /// Hard sigmoid: `clip(0.2 * x + 0.5, 0, 1)`.
    HardSigmoid,
}

impl Activation {
    /// Apply the activation to `xs`, returning a new tensor of the same shape.
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        match *self {
            Activation::Linear => Ok(xs.clone()),
            Activation::ReLU => xs.relu(),
            Activation::LeakyReLU(alpha) => {
                // max(x, alpha * x) equals x for x > 0 and alpha*x otherwise
                // (valid for 0 <= alpha <= 1, the usual leaky-ReLU regime).
                xs.maximum(&xs.affine(alpha, 0.0)?)
            }
            Activation::ELU(alpha) => xs.elu(alpha),
            Activation::Sigmoid => sigmoid(xs),
            Activation::Tanh => xs.tanh(),
            Activation::Softmax => softmax(xs, D::Minus1),
            Activation::LogSoftmax => log_softmax(xs, D::Minus1),
            Activation::GELU => xs.gelu(),
            Activation::SiLU => silu(xs),
            Activation::Softplus => softplus(xs),
            Activation::Mish => xs.mul(&softplus(xs)?.tanh()?),
            Activation::HardSigmoid => xs.affine(0.2, 0.5)?.clamp(0.0, 1.0),
        }
    }
}

/// `ln(1 + e^x)`, used by both Softplus and Mish.
fn softplus(xs: &Tensor) -> Result<Tensor> {
    (xs.exp()? + 1.0)?.log()
}

impl FromStr for Activation {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let act = match s.to_ascii_lowercase().as_str() {
            "linear" | "none" | "identity" | "" => Activation::Linear,
            "relu" => Activation::ReLU,
            "leaky_relu" | "leakyrelu" => Activation::LeakyReLU(0.01),
            "elu" => Activation::ELU(1.0),
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            "log_softmax" | "logsoftmax" => Activation::LogSoftmax,
            "gelu" => Activation::GELU,
            "silu" | "swish" => Activation::SiLU,
            "softplus" => Activation::Softplus,
            "mish" => Activation::Mish,
            "hard_sigmoid" | "hardsigmoid" => Activation::HardSigmoid,
            other => bail!(
                "Unknown activation `{other}`. Valid values: linear, relu, leaky_relu, \
                 elu, sigmoid, tanh, softmax, log_softmax, gelu, silu/swish, softplus, \
                 mish, hard_sigmoid"
            ),
        };
        Ok(act)
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Activation::Linear => "linear",
            Activation::ReLU => "relu",
            Activation::LeakyReLU(_) => "leaky_relu",
            Activation::ELU(_) => "elu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Softmax => "softmax",
            Activation::LogSoftmax => "log_softmax",
            Activation::GELU => "gelu",
            Activation::SiLU => "silu",
            Activation::Softplus => "softplus",
            Activation::Mish => "mish",
            Activation::HardSigmoid => "hard_sigmoid",
        };
        f.write_str(name)
    }
}

/// Padding strategy for convolution and pooling layers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Padding {
    /// No padding; output shrinks by `kernel - 1` (Keras `"valid"`).
    #[default]
    Valid,
    /// Zero-pad so the output keeps the input's spatial size for `stride == 1`
    /// (Keras `"same"`, for odd kernels).
    Same,
    /// Explicit symmetric padding of the given number of pixels per side.
    Explicit(usize),
}

impl Padding {
    /// Resolve the per-side padding in pixels for a given kernel size.
    pub fn to_pixels(self, kernel_size: usize) -> usize {
        match self {
            Padding::Valid => 0,
            Padding::Same => kernel_size / 2,
            Padding::Explicit(p) => p,
        }
    }
}

impl FromStr for Padding {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "valid" => Ok(Padding::Valid),
            "same" => Ok(Padding::Same),
            other => bail!("Unknown padding `{other}`. Valid values: valid, same"),
        }
    }
}

/// Weight-initialization strategies, mapped onto Candle's [`Init`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Initializer {
    /// All zeros.
    Zeros,
    /// All ones.
    Ones,
    /// A single constant value.
    Constant(f64),
    /// Normal distribution with the given mean / standard deviation.
    RandomNormal {
        /// Distribution mean.
        mean: f64,
        /// Distribution standard deviation.
        stdev: f64,
    },
    /// Uniform distribution on `[lo, up)`.
    RandomUniform {
        /// Lower bound.
        lo: f64,
        /// Upper bound.
        up: f64,
    },
    /// Glorot/Xavier uniform initialization.
    #[default]
    GlorotUniform,
    /// Glorot/Xavier normal initialization.
    GlorotNormal,
    /// He/Kaiming uniform initialization (good default for ReLU networks).
    HeUniform,
    /// He/Kaiming normal initialization.
    HeNormal,
}

impl Initializer {
    /// Convert to the Candle [`Init`] used by `VarBuilder::get_with_hints`.
    ///
    /// Glorot variants are approximated with fan-in Kaiming using a linear gain,
    /// which matches Xavier's scaling closely for the shapes used here.
    pub fn to_init(self) -> Init {
        match self {
            Initializer::Zeros => Init::Const(0.0),
            Initializer::Ones => Init::Const(1.0),
            Initializer::Constant(c) => Init::Const(c),
            Initializer::RandomNormal { mean, stdev } => Init::Randn { mean, stdev },
            Initializer::RandomUniform { lo, up } => Init::Uniform { lo, up },
            Initializer::GlorotUniform => Init::Kaiming {
                dist: NormalOrUniform::Uniform,
                fan: FanInOut::FanIn,
                non_linearity: NonLinearity::Linear,
            },
            Initializer::GlorotNormal => Init::Kaiming {
                dist: NormalOrUniform::Normal,
                fan: FanInOut::FanIn,
                non_linearity: NonLinearity::Linear,
            },
            Initializer::HeUniform => Init::Kaiming {
                dist: NormalOrUniform::Uniform,
                fan: FanInOut::FanIn,
                non_linearity: NonLinearity::ReLU,
            },
            Initializer::HeNormal => Init::Kaiming {
                dist: NormalOrUniform::Normal,
                fan: FanInOut::FanIn,
                non_linearity: NonLinearity::ReLU,
            },
        }
    }
}

/// Weight regularizers.
///
/// L2 is applied cheaply through the optimizer's `weight_decay`; L1 / elastic-net
/// penalties (which must be added to the loss) are represented here for API
/// completeness and are documented as a roadmap item.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Regularizer {
    /// L1 penalty with the given coefficient.
    L1(f64),
    /// L2 (weight-decay) penalty with the given coefficient.
    L2(f64),
    /// Combined L1 + L2 (elastic net).
    L1L2 {
        /// L1 coefficient.
        l1: f64,
        /// L2 coefficient.
        l2: f64,
    },
}

impl Regularizer {
    /// The L2 coefficient that can be forwarded to an optimizer as weight decay.
    pub fn l2_weight_decay(self) -> f64 {
        match self {
            Regularizer::L1(_) => 0.0,
            Regularizer::L2(l2) => l2,
            Regularizer::L1L2 { l2, .. } => l2,
        }
    }
}

/// Optimization algorithms with ergonomic, self-documenting constructors.
///
/// Prefer the constructors over the struct variants so you only specify the
/// hyper-parameters you care about:
///
/// ```
/// use kerasty::Optimizer;
/// let opt = Optimizer::adam(1e-3);          // sensible Adam defaults
/// let sgd = Optimizer::sgd(0.01);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Optimizer {
    /// Stochastic gradient descent.
    Sgd {
        /// Learning rate.
        lr: f64,
    },
    /// Adam / AdamW.
    Adam {
        /// Learning rate.
        lr: f64,
        /// First-moment decay.
        beta1: f64,
        /// Second-moment decay.
        beta2: f64,
        /// Numerical-stability epsilon.
        eps: f64,
        /// Decoupled weight decay (0 = plain Adam).
        weight_decay: f64,
    },
}

impl Optimizer {
    /// SGD with the given learning rate.
    pub fn sgd(lr: f64) -> Self {
        Optimizer::Sgd { lr }
    }

    /// Adam with Keras' default betas/epsilon and no weight decay.
    pub fn adam(lr: f64) -> Self {
        Optimizer::Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Adam with fully specified hyper-parameters (AdamW when `weight_decay > 0`).
    pub fn adam_with(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Optimizer::Adam {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        }
    }

    /// The configured learning rate.
    pub fn learning_rate(&self) -> f64 {
        match self {
            Optimizer::Sgd { lr } => *lr,
            Optimizer::Adam { lr, .. } => *lr,
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::adam(1e-3)
    }
}

/// Loss functions minimized during training and reported during evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    /// Mean squared error.
    MSE,
    /// Mean absolute error (L1).
    MAE,
    /// Huber loss with the given delta.
    Huber(f64),
    /// Negative log-likelihood (expects log-probabilities and class indices).
    NLL,
    /// Softmax cross-entropy from logits (expects class indices).
    CrossEntropy,
    /// Binary cross-entropy computed from logits (numerically stable).
    BinaryCrossEntropyWithLogit,
    /// Binary cross-entropy computed from probabilities in `(0, 1)`.
    BinaryCrossEntropy,
}

impl Loss {
    /// Compute the scalar loss between predictions and targets.
    pub fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        match *self {
            Loss::MSE => mse(pred, target),
            Loss::MAE => (pred - target)?.abs()?.mean_all(),
            Loss::Huber(delta) => huber(pred, target, delta),
            Loss::NLL => nll(pred, target),
            Loss::CrossEntropy => cross_entropy(pred, target),
            Loss::BinaryCrossEntropyWithLogit => binary_cross_entropy_with_logit(pred, target),
            Loss::BinaryCrossEntropy => {
                // -mean( t*log(p) + (1-t)*log(1-p) ), with clamping for stability.
                let eps = 1e-7;
                let p = pred.clamp(eps, 1.0 - eps)?;
                let term1 = target.mul(&p.log()?)?;
                let one_minus_t = target.affine(-1.0, 1.0)?;
                let one_minus_p = p.affine(-1.0, 1.0)?;
                let term2 = one_minus_t.mul(&one_minus_p.log()?)?;
                term1.add(&term2)?.neg()?.mean_all()
            }
        }
    }
}

/// Evaluation metrics reported by [`Model::evaluate`](crate::Model::evaluate).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    /// Mean squared error.
    MSE,
    /// Mean absolute error.
    MAE,
    /// Binary accuracy with a 0.5 decision threshold.
    Accuracy,
    /// Alias for [`Metric::Accuracy`] with explicit binary semantics.
    BinaryAccuracy,
    /// Categorical accuracy: `argmax(pred) == argmax(target)` over the last axis.
    CategoricalAccuracy,
}

impl Metric {
    /// The Keras-style name of the metric (used in `summary()` and logs).
    pub fn name(&self) -> &'static str {
        match self {
            Metric::MSE => "mse",
            Metric::MAE => "mae",
            Metric::Accuracy | Metric::BinaryAccuracy => "accuracy",
            Metric::CategoricalAccuracy => "categorical_accuracy",
        }
    }

    /// Compute the metric as a single `f64`, dtype-agnostically.
    pub fn compute(&self, pred: &Tensor, target: &Tensor) -> Result<f64> {
        let pred = pred.to_dtype(DType::F64)?;
        let target = target.to_dtype(DType::F64)?;
        match self {
            Metric::MSE => mse(&pred, &target)?.to_scalar::<f64>(),
            Metric::MAE => (&pred - &target)?.abs()?.mean_all()?.to_scalar::<f64>(),
            Metric::Accuracy | Metric::BinaryAccuracy => {
                let p = pred.flatten_all()?.to_vec1::<f64>()?;
                let t = target.flatten_all()?.to_vec1::<f64>()?;
                let correct = p
                    .iter()
                    .zip(t.iter())
                    .filter(|(&pi, &ti)| (pi >= 0.5) == (ti >= 0.5))
                    .count();
                Ok(correct as f64 / p.len().max(1) as f64)
            }
            Metric::CategoricalAccuracy => {
                let pred_idx = pred.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
                let target_idx = target.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
                let correct = pred_idx
                    .iter()
                    .zip(target_idx.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                Ok(correct as f64 / pred_idx.len().max(1) as f64)
            }
        }
    }
}
