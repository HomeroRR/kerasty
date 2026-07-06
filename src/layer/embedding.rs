//! Embedding lookup layer.

use candle_core::{bail, Result, Tensor};
use candle_nn::{embedding, Embedding as CandleEmbedding, Module, VarBuilder};

use crate::common::traits::Layer;

/// Maps integer token indices to dense vectors.
///
/// The input is an integer tensor (dtype `u32`) of shape `(batch, seq_len)`; the
/// output is `(batch, seq_len, output_dim)`.
#[derive(Clone, Debug)]
pub struct Embedding {
    input_dim: usize,
    output_dim: usize,
    inner: Option<CandleEmbedding>,
}

impl Embedding {
    /// Create an embedding table of `input_dim` entries (vocabulary size), each
    /// an `output_dim`-dimensional vector.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Embedding {
            input_dim,
            output_dim,
            inner: None,
        }
    }
}

impl Layer for Embedding {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        self.inner = Some(embedding(self.input_dim, self.output_dim, vb)?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let Some(emb) = &self.inner else {
            bail!("Embedding layer used before `build`/`compile`");
        };
        emb.forward(xs)
    }

    fn kind(&self) -> &'static str {
        "Embedding"
    }
}
