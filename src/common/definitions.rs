use candle_core::{bail, Error, Result};
use candle_nn::{AdamW, SGD};
use std::str::FromStr;
/* Define the Activation types */
#[derive(Clone, Debug)]
pub enum Activation {
    Linear,
    ReLU,
    Softmax,
    Sigmoid,
}

impl FromStr for Activation {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "linear" => Ok(Activation::Linear),
            "relu" => Ok(Activation::ReLU),
            "softmax" => Ok(Activation::Softmax),
            "sigmoid" => Ok(Activation::Sigmoid),
            _ => bail!("Unknown Activation type: {}", s),
        }
    }
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
    Adam(f64, f64, f64, f64, f64),
}
pub enum OptimizerInstance {
    SGD(SGD),
    Adam(AdamW),
}

/* Define the Loss types */
#[derive(Clone, Debug)]
pub enum Loss {
    MSE,
    NLL,
    BinaryCrossEntropyWithLogit,
    CrossEntropy,
}

/* Define the Metric types */
#[derive(Clone, Debug)]
pub enum Metric {
    MSE,
    Accuracy,
}
