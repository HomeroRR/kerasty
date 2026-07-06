//! Recurrent layers: `SimpleRNN`, `LSTM` and `GRU`.
//!
//! All three consume a sequence shaped `(batch, seq_len, features)`. By default
//! they return only the last time step's hidden state — shape `(batch, units)`,
//! matching Keras' `return_sequences=False`. Set `.with_return_sequences(true)`
//! to return the full sequence of hidden states, `(batch, seq_len, units)`.

use candle_core::{bail, IndexOp, Result, Tensor};
use candle_nn::{
    gru, linear, linear_no_bias, lstm, GRUConfig, LSTMConfig, Linear, Module, VarBuilder, GRU,
    LSTM, RNN,
};

use crate::common::definitions::Activation;
use crate::common::traits::Layer;

/// Long Short-Term Memory recurrent layer.
#[derive(Clone, Debug)]
pub struct Lstm {
    input_dim: usize,
    units: usize,
    return_sequences: bool,
    inner: Option<LSTM>,
}

impl Lstm {
    /// Create an LSTM mapping `input_dim` features to `units` hidden units.
    pub fn new(input_dim: usize, units: usize) -> Self {
        Lstm {
            input_dim,
            units,
            return_sequences: false,
            inner: None,
        }
    }

    /// Builder: return every time step instead of only the last.
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }
}

impl Layer for Lstm {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        self.inner = Some(lstm(self.input_dim, self.units, LSTMConfig::default(), vb)?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let Some(cell) = &self.inner else {
            bail!("LSTM layer used before `build`/`compile`");
        };
        let states = cell.seq(xs)?;
        finish_states(cell, &states, self.return_sequences)
    }

    fn kind(&self) -> &'static str {
        "LSTM"
    }
}

/// Gated Recurrent Unit recurrent layer.
#[derive(Clone, Debug)]
pub struct Gru {
    input_dim: usize,
    units: usize,
    return_sequences: bool,
    inner: Option<GRU>,
}

impl Gru {
    /// Create a GRU mapping `input_dim` features to `units` hidden units.
    pub fn new(input_dim: usize, units: usize) -> Self {
        Gru {
            input_dim,
            units,
            return_sequences: false,
            inner: None,
        }
    }

    /// Builder: return every time step instead of only the last.
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }
}

impl Layer for Gru {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        self.inner = Some(gru(self.input_dim, self.units, GRUConfig::default(), vb)?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let Some(cell) = &self.inner else {
            bail!("GRU layer used before `build`/`compile`");
        };
        let states = cell.seq(xs)?;
        finish_states(cell, &states, self.return_sequences)
    }

    fn kind(&self) -> &'static str {
        "GRU"
    }
}

/// Convert a sequence of recurrent states either to the full `(batch, seq, units)`
/// tensor or to just the last step `(batch, units)`.
fn finish_states<R: RNN>(cell: &R, states: &[R::State], return_sequences: bool) -> Result<Tensor> {
    if return_sequences {
        cell.states_to_tensor(states)
    } else {
        // The last state stacked as a length-1 sequence, then squeezed.
        let last = &states[states.len() - 1];
        cell.states_to_tensor(std::slice::from_ref(last))?
            .squeeze(1)
    }
}

/// A vanilla (Elman) recurrent layer: `h_t = act(x_t·Wᵢ + h_{t-1}·Wₕ + b)`.
///
/// Candle ships LSTM/GRU but not a plain RNN cell, so this is implemented
/// directly from two linear maps.
#[derive(Clone, Debug)]
pub struct SimpleRNN {
    input_dim: usize,
    units: usize,
    activation: Activation,
    return_sequences: bool,
    input_kernel: Option<Linear>,
    recurrent_kernel: Option<Linear>,
}

impl SimpleRNN {
    /// Create a SimpleRNN mapping `input_dim` features to `units` hidden units
    /// (default `tanh` activation).
    pub fn new(input_dim: usize, units: usize) -> Self {
        SimpleRNN {
            input_dim,
            units,
            activation: Activation::Tanh,
            return_sequences: false,
            input_kernel: None,
            recurrent_kernel: None,
        }
    }

    /// Builder: override the activation (default `tanh`).
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Builder: return every time step instead of only the last.
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }
}

impl Layer for SimpleRNN {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        self.input_kernel = Some(linear(self.input_dim, self.units, vb.pp("input"))?);
        self.recurrent_kernel = Some(linear_no_bias(self.units, self.units, vb.pp("recurrent"))?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let (Some(wi), Some(wh)) = (&self.input_kernel, &self.recurrent_kernel) else {
            bail!("SimpleRNN layer used before `build`/`compile`");
        };
        let (batch, seq_len, _features) = xs.dims3()?;
        let mut h = Tensor::zeros((batch, self.units), xs.dtype(), xs.device())?;
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = xs.i((.., t, ..))?.contiguous()?;
            let pre = wi.forward(&x_t)?.add(&wh.forward(&h)?)?;
            h = self.activation.apply(&pre)?;
            outputs.push(h.clone());
        }
        if self.return_sequences {
            Tensor::stack(&outputs, 1)
        } else {
            Ok(h)
        }
    }

    fn kind(&self) -> &'static str {
        "SimpleRNN"
    }
}
