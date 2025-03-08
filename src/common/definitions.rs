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