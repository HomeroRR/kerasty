//! Spatial pooling layers (2D). These have no trainable parameters.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::common::traits::Layer;

/// Max pooling over 2D inputs shaped `(batch, channels, height, width)`.
#[derive(Clone, Debug)]
pub struct MaxPool2D {
    pool_size: usize,
    stride: usize,
}

impl MaxPool2D {
    /// Create a max-pool layer with a square window; the stride defaults to the
    /// window size (non-overlapping), as in Keras.
    pub fn new(pool_size: usize) -> Self {
        MaxPool2D {
            pool_size,
            stride: pool_size,
        }
    }

    /// Builder: override the stride.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }
}

impl Layer for MaxPool2D {
    fn build(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        xs.max_pool2d_with_stride(self.pool_size, self.stride)
    }

    fn kind(&self) -> &'static str {
        "MaxPool2D"
    }
}

/// Average pooling over 2D inputs shaped `(batch, channels, height, width)`.
#[derive(Clone, Debug)]
pub struct AvgPool2D {
    pool_size: usize,
    stride: usize,
}

impl AvgPool2D {
    /// Create an average-pool layer with a square window; the stride defaults to
    /// the window size.
    pub fn new(pool_size: usize) -> Self {
        AvgPool2D {
            pool_size,
            stride: pool_size,
        }
    }

    /// Builder: override the stride.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }
}

impl Layer for AvgPool2D {
    fn build(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.pool_size, self.stride)
    }

    fn kind(&self) -> &'static str {
        "AvgPool2D"
    }
}

/// Global average pooling: reduces `(batch, channels, height, width)` to
/// `(batch, channels)` by averaging each channel's spatial map.
#[derive(Clone, Debug, Default)]
pub struct GlobalAveragePooling2D;

impl GlobalAveragePooling2D {
    /// Create a global average pooling layer.
    pub fn new() -> Self {
        GlobalAveragePooling2D
    }
}

impl Layer for GlobalAveragePooling2D {
    fn build(&mut self, _vb: VarBuilder) -> Result<()> {
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        // Mean over the height and width axes (2 and 3).
        xs.mean(candle_core::D::Minus1)?
            .mean(candle_core::D::Minus1)
    }

    fn kind(&self) -> &'static str {
        "GlobalAveragePooling2D"
    }
}
