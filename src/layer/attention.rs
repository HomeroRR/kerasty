//! Multi-head self-attention — the core primitive of Transformer models.
//!
//! With `Embedding` + `LayerNorm` + `Dense`, this is enough to compose encoder
//! (BERT-style) and decoder (Llama-style) blocks in user code.

use candle_core::{bail, Result, Tensor, D};
use candle_nn::{linear, ops::softmax, Linear, Module, VarBuilder};

use crate::common::traits::Layer;

/// Multi-head scaled dot-product self-attention over `(batch, seq_len, embed_dim)`.
///
/// `embed_dim` must be divisible by `num_heads`.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: Option<Linear>,
    k_proj: Option<Linear>,
    v_proj: Option<Linear>,
    out_proj: Option<Linear>,
}

impl MultiHeadAttention {
    /// Create a multi-head attention layer.
    ///
    /// # Panics
    /// Panics if `embed_dim` is not divisible by `num_heads`.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        );
        MultiHeadAttention {
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            out_proj: None,
        }
    }
}

impl Layer for MultiHeadAttention {
    fn build(&mut self, vb: VarBuilder) -> Result<()> {
        let e = self.embed_dim;
        self.q_proj = Some(linear(e, e, vb.pp("q_proj"))?);
        self.k_proj = Some(linear(e, e, vb.pp("k_proj"))?);
        self.v_proj = Some(linear(e, e, vb.pp("v_proj"))?);
        self.out_proj = Some(linear(e, e, vb.pp("out_proj"))?);
        Ok(())
    }

    fn forward(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let (Some(wq), Some(wk), Some(wv), Some(wo)) =
            (&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj)
        else {
            bail!("MultiHeadAttention layer used before `build`/`compile`");
        };
        let (b, s, e) = xs.dims3()?;
        let (h, d) = (self.num_heads, self.head_dim);

        // Project, then split into heads: (b, s, e) -> (b, h, s, d).
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()
        };
        let q = split(wq.forward(xs)?)?;
        let k = split(wk.forward(xs)?)?;
        let v = split(wv.forward(xs)?)?;

        // Scaled dot-product attention.
        let scale = 1.0 / (d as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let weights = softmax(&scores, D::Minus1)?;
        let context = weights.matmul(&v)?; // (b, h, s, d)

        // Merge heads back: (b, h, s, d) -> (b, s, e), then output projection.
        let merged = context.transpose(1, 2)?.contiguous()?.reshape((b, s, e))?;
        wo.forward(&merged)
    }

    fn kind(&self) -> &'static str {
        "MultiHeadAttention"
    }
}
