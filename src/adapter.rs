//! Adapters: small trained deltas over a base model.
//!
//! A base model is expensive to train and belongs to a family. An
//! adapter is cheap: two thin matrices per attention projection,
//! trained on a few glyphs with the base frozen, that add `B·A` to
//! the projection's weight. That is LoRA (Hu et al., 2021), with the
//! shape ComfyUI's `LoraLoader` gives it: a directory, a strength, and
//! the rule that two applied in a row both take.
//!
//! An adapter directory holds `adapter.json` (rank, alpha, the base
//! it was trained over, the projections it touches) and
//! `adapter.safetensors` with tensors named after the base's, plus
//! `.lora_a` and `.lora_b`. Applying one is arithmetic on the base
//! weights at load time, so inference is exactly as fast as without.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The projections an adapter can touch, as their tensor names end.
pub const TARGETS: &[&str] = &["query_proj", "key_proj", "value_proj", "out_proj"];

/// `adapter.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// The rank of the update.
    pub rank: usize,
    /// The scale numerator: the update is `alpha / rank` times `B·A`.
    pub alpha: f64,
    /// The base model's directory name, for the reader.
    #[serde(default)]
    pub base: String,
    /// Which projections carry a delta.
    #[serde(default = "all_targets")]
    pub targets: Vec<String>,
    /// How many blocks the base has, so a mismatch is caught early.
    #[serde(default)]
    pub layers: usize,
}

fn all_targets() -> Vec<String> {
    TARGETS.iter().map(|t| (*t).to_string()).collect()
}

impl AdapterConfig {
    /// The multiplier on `B·A`.
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank.max(1) as f64
    }
}

/// An adapter on disk, read but not applied.
#[derive(Debug, Clone)]
pub struct Adapter {
    pub dir: PathBuf,
    pub config: AdapterConfig,
}

impl Adapter {
    /// Reads a directory. Does not touch the weights.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let config_path = dir.join("adapter.json");
        if !config_path.is_file() {
            return Err(Error::MissingFile(dir, "adapter.json"));
        }
        let text = std::fs::read_to_string(&config_path).map_err(|source| Error::Io {
            path: config_path.clone(),
            source,
        })?;
        let config: AdapterConfig =
            serde_json::from_str(&text).map_err(|source| Error::Config {
                path: config_path,
                source,
            })?;
        if !dir.join("adapter.safetensors").is_file() {
            return Err(Error::MissingFile(dir, "adapter.safetensors"));
        }
        Ok(Self { dir, config })
    }

    pub fn weights_path(&self) -> PathBuf {
        self.dir.join("adapter.safetensors")
    }

    /// The tensor names an adapter over `layers` blocks carries, in
    /// pairs: `(a, b)` for each block and target.
    pub fn tensor_names(layers: usize, targets: &[String]) -> Vec<(String, String)> {
        let mut names = Vec::new();
        for i in 0..layers {
            for t in targets {
                let stem = format!("blocks.{i}.attn.{t}");
                names.push((format!("{stem}.lora_a"), format!("{stem}.lora_b")));
            }
        }
        names
    }

    /// Adds `strength` times this adapter's update to the base weights
    /// in place: `W += strength * scale * B·A` for every projection it
    /// carries. A second call with another adapter stacks.
    pub fn merge_into(
        &self,
        weights: &mut HashMap<String, Tensor>,
        strength: f64,
        device: &Device,
    ) -> Result<usize> {
        let deltas = candle_core::safetensors::load(self.weights_path(), device)?;
        let factor = strength * self.config.scale();
        let mut applied = 0;
        for (a_name, b_name) in Self::tensor_names(self.config.layers.max(1), &self.config.targets)
        {
            let (Some(a), Some(b)) = (deltas.get(&a_name), deltas.get(&b_name)) else {
                continue;
            };
            let weight_name = format!("{}.weight", a_name.trim_end_matches(".lora_a"));
            let Some(w) = weights.get(&weight_name) else {
                continue;
            };
            let update = (b.matmul(a)? * factor)?;
            let merged = w.broadcast_add(&update)?;
            weights.insert(weight_name, merged);
            applied += 1;
        }
        Ok(applied)
    }
}

/// A runtime LoRA on one projection, for training: `x·Aᵀ·Bᵀ`
/// scaled, added to the frozen projection's output.
#[derive(Debug, Clone)]
pub struct Lora {
    /// `rank × in`.
    pub a: Tensor,
    /// `out × rank`.
    pub b: Tensor,
    /// `alpha / rank`.
    pub scale: f64,
}

impl Lora {
    /// The update for an activation `x` of shape `(.., in)`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let a_t = self.a.t()?;
        let b_t = self.b.t()?;
        let h = x.broadcast_matmul(&a_t)?;
        Ok((h.broadcast_matmul(&b_t)? * self.scale)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_base_layout() {
        let names = Adapter::tensor_names(2, &["query_proj".to_string()]);
        assert_eq!(names.len(), 2);
        assert_eq!(names[1].0, "blocks.1.attn.query_proj.lora_a");
        assert_eq!(names[1].1, "blocks.1.attn.query_proj.lora_b");
    }

    #[test]
    fn a_zero_b_is_no_change() {
        let dev = Device::Cpu;
        let x = Tensor::ones((3, 4), candle_core::DType::F32, &dev).unwrap();
        let lora = Lora {
            a: Tensor::ones((2, 4), candle_core::DType::F32, &dev).unwrap(),
            b: Tensor::zeros((5, 2), candle_core::DType::F32, &dev).unwrap(),
            scale: 2.0,
        };
        let y = lora.apply(&x).unwrap();
        assert_eq!(y.dims(), &[3, 5]);
        let sum: f32 = y.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn scale_is_alpha_over_rank() {
        let cfg = AdapterConfig {
            rank: 8,
            alpha: 16.0,
            base: String::new(),
            targets: all_targets(),
            layers: 6,
        };
        assert_eq!(cfg.scale(), 2.0);
    }
}

#[cfg(test)]
mod train_probe {
    use super::*;
    use candle_core::DType;
    use candle_nn::{Optimizer, VarBuilder, VarMap};

    /// Every variable, base and adapter, must get a gradient, and one
    /// step must move the adapter's B. This is the check that caught
    /// candle-nn's fused norm and softmax cutting the graph.
    #[test]
    fn a_step_moves_b() {
        let dev = Device::Cpu;
        let cfg = crate::checkpoint::ModelConfig {
            kind: crate::checkpoint::ModelKind::Outline,
            dims: 16,
            layers: 1,
            heads: 2,
            vocab_size: 0,
            max_len: 32,
            delta_center: None,
            trim_close: false,
            extra: Default::default(),
        };
        let vocab = crate::tokenizer::Vocab::new(vec!["a".into()], vec![]);
        let base = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&base, DType::F32, &dev);
        let mut model =
            crate::outline::OutlineModel::build(base_vb, &cfg, vocab, dev.clone(), 0.0).unwrap();
        let lora_map = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&lora_map, DType::F32, &dev);
        model.attach_lora(&lora_vb, 4, 8.0, 16).unwrap();
        let vars = lora_map.all_vars();
        assert_eq!(vars.len(), 8);
        let before: Vec<f32> = vars
            .iter()
            .map(|v| {
                v.as_tensor()
                    .abs()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            })
            .collect();
        let ids = Tensor::new(&[[1u32, 5, 6, 7, 2, 0]], &dev).unwrap();
        let logits = model.forward_batch(&ids, true).unwrap();
        let loss = logits.sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let with_grad = vars.iter().filter(|v| grads.get(v).is_some()).count();
        let base_vars = base.all_vars();
        let base_with_grad = base_vars.iter().filter(|v| grads.get(v).is_some()).count();
        let mut opt = candle_nn::AdamW::new(
            lora_map.all_vars(),
            candle_nn::ParamsAdamW {
                lr: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        opt.backward_step(&loss).unwrap();
        let after: Vec<f32> = vars
            .iter()
            .map(|v| {
                v.as_tensor()
                    .abs()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            })
            .collect();
        assert_eq!(with_grad, 8, "every adapter matrix gets a gradient");
        assert_eq!(
            base_with_grad,
            base_vars.len(),
            "every base variable gets a gradient"
        );
        let moved = before
            .iter()
            .zip(&after)
            .filter(|(b, a)| (*b - *a).abs() > 1e-6)
            .count();
        assert!(
            moved > 0,
            "no adapter variable moved: before {before:?} after {after:?}"
        );
        // And the forward must differ now.
        let logits2 = model.forward_batch(&ids, false).unwrap();
        let diff: f32 = (logits2 - logits)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff > 1e-6, "forward unchanged after a step");
    }
}
