//! The outline model: a small decoder-only transformer over drawing
//! tokens.
//!
//! Architecture is deliberately plain, because the checkpoints were
//! trained that way: learned token and position embeddings, pre-norm
//! blocks with multi-head attention and a 4x GELU feed-forward, a
//! final norm, and a linear head. Tensor names follow the training
//! lab's, so a checkpoint loads without renaming.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim,
    Embedding, LayerNorm, Linear, Module, VarBuilder,
};

use crate::checkpoint::{Checkpoint, ModelKind};
use crate::error::Result;
use crate::tokenizer::Vocab;

struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
}

impl Block {
    fn load(vb: VarBuilder, dims: usize, heads: usize) -> Result<Self> {
        let attn = vb.pp("attn");
        Ok(Self {
            // The lab's attention projections carry no bias.
            q: linear_no_bias(dims, dims, attn.pp("query_proj"))?,
            k: linear_no_bias(dims, dims, attn.pp("key_proj"))?,
            v: linear_no_bias(dims, dims, attn.pp("value_proj"))?,
            out: linear_no_bias(dims, dims, attn.pp("out_proj"))?,
            norm1: layer_norm(dims, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(dims, 1e-5, vb.pp("norm2"))?,
            fc1: linear(dims, 4 * dims, vb.pp("mlp.layers.0"))?,
            fc2: linear(4 * dims, dims, vb.pp("mlp.layers.2"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (seq, dims) = x.dims2()?;
        let head_dim = dims / self.heads;

        let h = self.norm1.forward(x)?;
        let split = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((seq, self.heads, head_dim))?
                .transpose(0, 1)?
                .contiguous()?)
        };
        let q = split(self.q.forward(&h)?)?;
        let k = split(self.k.forward(&h)?)?;
        let v = split(self.v.forward(&h)?)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let attn = softmax_last_dim(&scores)?;
        let context = attn
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((seq, dims))?
            .contiguous()?;
        let x = (x + self.out.forward(&context)?)?;

        let h = self.norm2.forward(&x)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu_erf()?)?;
        Ok((x + h)?)
    }
}

/// A loaded outline model, ready to run.
pub struct OutlineModel {
    tok: Embedding,
    pos: Embedding,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    device: Device,
    max_len: usize,
    vocab: Vocab,
}

impl OutlineModel {
    /// Load the weights. Reads about 50MB from disk for a 12M model.
    pub fn load(checkpoint: &Checkpoint) -> Result<Self> {
        checkpoint.require(ModelKind::Outline)?;
        let cfg = &checkpoint.config;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[checkpoint.weights_path()],
                DType::F32,
                &device,
            )?
        };
        let vocab = Vocab::new(
            checkpoint.glyph_names.clone(),
            checkpoint.unicodes.clone(),
        );
        let vocab_size =
            if cfg.vocab_size > 0 { cfg.vocab_size } else { vocab.len() };

        let blocks = (0..cfg.layers)
            .map(|i| {
                Block::load(vb.pp(format!("blocks.{i}")), cfg.dims, cfg.heads)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            tok: embedding(vocab_size, cfg.dims, vb.pp("tok"))?,
            pos: embedding(cfg.max_len, cfg.dims, vb.pp("pos"))?,
            blocks,
            norm: layer_norm(cfg.dims, 1e-5, vb.pp("norm"))?,
            head: linear(cfg.dims, vocab_size, vb.pp("out"))?,
            device,
            max_len: cfg.max_len,
            vocab,
        })
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Logits for every position in the sequence.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let seq = ids.len().min(self.max_len);
        let ids = Tensor::new(&ids[..seq], &self.device)?;
        let positions =
            Tensor::arange(0u32, seq as u32, &self.device)?;
        let mut h =
            (self.tok.forward(&ids)? + self.pos.forward(&positions)?)?;
        let mask = causal_mask(seq, &self.device)?;
        for block in &self.blocks {
            h = block.forward(&h, &mask)?;
        }
        Ok(self.head.forward(&self.norm.forward(&h)?)?)
    }

    /// The most likely next token after this sequence.
    ///
    /// Greedy on purpose: a font wants the same answer twice, and a
    /// designer reviewing a proposal should not have to wonder whether
    /// a rerun would have been better.
    pub fn next_token(&self, ids: &[u32]) -> Result<u32> {
        let logits = self.forward(ids)?;
        let last = logits.i(logits.dim(0)? - 1)?;
        Ok(last.argmax(D::Minus1)?.to_scalar::<u32>()?)
    }
}

/// Additive mask that stops a position attending to later ones.
fn causal_mask(seq: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq * seq];
    for row in 0..seq {
        for col in (row + 1)..seq {
            data[row * seq + col] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(data, (seq, seq), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_hides_the_future_and_nothing_else() {
        let m = causal_mask(3, &Device::Cpu).unwrap();
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v[0], 0.0);
        assert!(v[1].is_infinite() && v[1] < 0.0);
        assert_eq!(v[3], 0.0);
        assert_eq!(v[4], 0.0);
        assert!(v[5].is_infinite());
    }
}
