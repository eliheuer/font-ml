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
    embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim, Dropout, Embedding,
    LayerNorm, Linear, Module, VarBuilder,
};

use crate::adapter::{Adapter, Lora};
use crate::checkpoint::{Checkpoint, ModelConfig, ModelKind};
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
    /// Only acts while training; inference passes `train = false`.
    drop: Dropout,
    /// Trainable adapters on q, k, v, out, when an adapter is being
    /// trained. Applied ones are merged into the weights instead.
    lora: [Option<Lora>; 4],
}

impl Block {
    fn load(vb: VarBuilder, dims: usize, heads: usize, dropout: f64) -> Result<Self> {
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
            drop: Dropout::new(dropout as f32),
            lora: [None, None, None, None],
        })
    }

    /// A projection, plus its adapter when training one.
    fn proj(&self, which: usize, x: &Tensor) -> Result<Tensor> {
        let linear = match which {
            0 => &self.q,
            1 => &self.k,
            2 => &self.v,
            _ => &self.out,
        };
        let y = linear.forward(x)?;
        match &self.lora[which] {
            Some(l) => Ok((y + l.apply(x)?)?),
            None => Ok(y),
        }
    }

    /// The same block over a batch: `x` is `(batch, seq, dims)`.
    fn forward_batch(&self, x: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
        let (b, seq, dims) = x.dims3()?;
        let head_dim = dims / self.heads;
        let h = self.norm1.forward(x)?;
        let split = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, seq, self.heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = split(self.proj(0, &h)?)?;
        let k = split(self.proj(1, &h)?)?;
        let v = split(self.proj(2, &h)?)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let attn = softmax_last_dim(&scores)?;
        let context = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq, dims))?
            .contiguous()?;
        let x = (x + self.drop.forward(&self.proj(3, &context)?, train)?)?;
        let h = self.norm2.forward(&x)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu_erf()?)?;
        Ok((x + self.drop.forward(&h, train)?)?)
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
        let q = split(self.proj(0, &h)?)?;
        let k = split(self.proj(1, &h)?)?;
        let v = split(self.proj(2, &h)?)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let attn = softmax_last_dim(&scores)?;
        let context = attn
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((seq, dims))?
            .contiguous()?;
        let x = (x + self.proj(3, &context)?)?;

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
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[checkpoint.weights_path()], DType::F32, &device)?
        };
        let vocab = Vocab::new(checkpoint.glyph_names.clone(), checkpoint.unicodes.clone());
        Self::build(vb, &checkpoint.config, vocab, device, 0.0)
    }

    /// Load the weights with adapters merged in, each at a strength.
    /// The order is the order given; each adds to what is there.
    pub fn load_with_adapters(
        checkpoint: &Checkpoint,
        adapters: &[(Adapter, f64)],
    ) -> Result<Self> {
        if adapters.is_empty() {
            return Self::load(checkpoint);
        }
        checkpoint.require(ModelKind::Outline)?;
        let device = Device::Cpu;
        let mut weights = candle_core::safetensors::load(checkpoint.weights_path(), &device)?;
        for (adapter, strength) in adapters {
            adapter.merge_into(&mut weights, *strength, &device)?;
        }
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let vocab = Vocab::new(checkpoint.glyph_names.clone(), checkpoint.unicodes.clone());
        Self::build(vb, &checkpoint.config, vocab, device, 0.0)
    }

    /// Attaches trainable adapters to every block's projections, from
    /// a variable source of the adapter's own. Base weights stay as
    /// they were loaded, so only the adapter learns.
    pub fn attach_lora(
        &mut self,
        vb: &VarBuilder,
        rank: usize,
        alpha: f64,
        dims: usize,
    ) -> Result<()> {
        use candle_nn::Init;
        let scale = alpha / rank.max(1) as f64;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            for (which, target) in crate::adapter::TARGETS.iter().enumerate() {
                let stem = vb.pp(format!("blocks.{i}.attn.{target}"));
                // A starts small and random, B at zero, so the model
                // begins exactly where the base is.
                let a = stem.get_with_hints(
                    (rank, dims),
                    "lora_a",
                    Init::Uniform {
                        lo: -1.0 / (dims as f64).sqrt(),
                        up: 1.0 / (dims as f64).sqrt(),
                    },
                )?;
                let b = stem.get_with_hints((dims, rank), "lora_b", Init::Const(0.0))?;
                block.lora[which] = Some(Lora { a, b, scale });
            }
        }
        Ok(())
    }

    /// Build the model over any variable source: a checkpoint on disk,
    /// or a fresh `VarMap` to train. Tensor names are the lab's either
    /// way.
    pub fn build(
        vb: VarBuilder,
        cfg: &ModelConfig,
        vocab: Vocab,
        device: Device,
        dropout: f64,
    ) -> Result<Self> {
        let vocab_size = if cfg.vocab_size > 0 {
            cfg.vocab_size
        } else {
            vocab.len()
        };
        let blocks = (0..cfg.layers)
            .map(|i| Block::load(vb.pp(format!("blocks.{i}")), cfg.dims, cfg.heads, dropout))
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
        let positions = Tensor::arange(0u32, seq as u32, &self.device)?;
        let mut h = (self.tok.forward(&ids)? + self.pos.forward(&positions)?)?;
        let mask = causal_mask(seq, &self.device)?;
        for block in &self.blocks {
            h = block.forward(&h, &mask)?;
        }
        Ok(self.head.forward(&self.norm.forward(&h)?)?)
    }

    /// Logits for a batch of sequences, `(batch, seq)` of token ids in,
    /// `(batch, seq, vocab)` out. `train` turns dropout on.
    pub fn forward_batch(&self, ids: &Tensor, train: bool) -> Result<Tensor> {
        let (_, seq) = ids.dims2()?;
        let positions = Tensor::arange(0u32, seq as u32, &self.device)?;
        let mut h = self
            .tok
            .forward(ids)?
            .broadcast_add(&self.pos.forward(&positions)?)?;
        let mask = causal_mask(seq, &self.device)?;
        for block in &self.blocks {
            h = block.forward_batch(&h, &mask, train)?;
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
