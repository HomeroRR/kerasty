//! Convolution layers (1D and 2D).

use candle_core::{bail, Result, Tensor};
use candle_nn::{conv1d, conv2d, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Module, VarBuilder};

use crate::common::definitions::{Activation, Padding};
use crate::common::traits::Layer;

/// 2D convolution over inputs shaped `(batch, in_channels, height, width)`.
///
/// ```
/// use kerasty::{Conv2D, Padding};
/// // 32 filters, 3x3 kernel over 3-channel input, ReLU, "same" padding
/// let conv = Conv2D::new(3, 32, 3, "relu").with_padding(Padding::Same);
/// ```
#[derive(Clone, Debug)]
pub struct Conv2D {
    in_channels: usize,
    filters: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: Padding,
    activation: Activation,
    conv: Option<Conv2d>,
}

impl Conv2D {
    /// Create a 2D convolution with `filters` output channels and a square
    /// `kernel_size`, applying the named `activation`.
    pub fn new(in_channels: usize, filters: usize, kernel_size: usize, activation: &str) -> Self {
        Conv2D {
            in_channels,
            filters,
            kernel_size,
            stride: 1,
            dilation: 1,
            padding: Padding::Valid,
            activation: activation.parse().unwrap_or_default(),
            conv: None,
        }
    }

    /// Builder: set the stride (default 1).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Builder: set the dilation (default 1).
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Builder: set the padding strategy (default [`Padding::Valid`]).
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }
}

impl Layer for Conv2D {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        let cfg = Conv2dConfig {
            padding: self.padding.to_pixels(self.kernel_size),
            stride: self.stride,
            dilation: self.dilation,
            ..Default::default()
        };
        self.conv = Some(conv2d(
            self.in_channels,
            self.filters,
            self.kernel_size,
            cfg,
            vb,
        )?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let Some(conv) = &self.conv else {
            bail!("Conv2D layer used before `build`/`compile`");
        };
        self.activation.apply(&conv.forward(xs)?)
    }

    fn kind(&self) -> &'static str {
        "Conv2D"
    }
}

/// 1D convolution over inputs shaped `(batch, in_channels, length)`.
#[derive(Clone, Debug)]
pub struct Conv1D {
    in_channels: usize,
    filters: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: Padding,
    activation: Activation,
    conv: Option<Conv1d>,
}

impl Conv1D {
    /// Create a 1D convolution with `filters` output channels and `kernel_size`.
    pub fn new(in_channels: usize, filters: usize, kernel_size: usize, activation: &str) -> Self {
        Conv1D {
            in_channels,
            filters,
            kernel_size,
            stride: 1,
            dilation: 1,
            padding: Padding::Valid,
            activation: activation.parse().unwrap_or_default(),
            conv: None,
        }
    }

    /// Builder: set the stride (default 1).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Builder: set the dilation (default 1).
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Builder: set the padding strategy (default [`Padding::Valid`]).
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }
}

impl Layer for Conv1D {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        let cfg = Conv1dConfig {
            padding: self.padding.to_pixels(self.kernel_size),
            stride: self.stride,
            dilation: self.dilation,
            ..Default::default()
        };
        self.conv = Some(conv1d(
            self.in_channels,
            self.filters,
            self.kernel_size,
            cfg,
            vb,
        )?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let Some(conv) = &self.conv else {
            bail!("Conv1D layer used before `build`/`compile`");
        };
        self.activation.apply(&conv.forward(xs)?)
    }

    fn kind(&self) -> &'static str {
        "Conv1D"
    }
}
