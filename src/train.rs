//! Training the outline model from two masters.
//!
//! A port of the training lab's `glyphlab.train` and `glyphlab.dataset`
//! to candle, so a model can be trained where it is used, with no
//! Python beside it. The corpus is two masters of one family: the
//! lighter is the base, the heavier the target. Each sample is one of
//! three sequences, chosen at random per step:
//!
//! - an interleaved pair (`PAIR`): the base outline with the target's
//!   offsets after every point, which is what boldening predicts;
//! - a weight-labelled single at an interpolated weight, when the
//!   glyph is drawn in both;
//! - the base glyph alone as a `W400` single, when it is not.
//!
//! Augmentation is the lab's: each contour starts at a random point,
//! and one name in ten becomes `UNK`. A fixed set of glyphs is held
//! out as validation, and the checkpoint with the lowest validation
//! loss is the one written.
//!
//! The lab's other corpora (OFL pretraining, sketch pairs) are not
//! here. This trains one family's weight transfer, which is the
//! workshop's job and the demo's.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::checkpoint::{Checkpoint, ModelConfig, ModelKind};
use crate::error::{Error, Result};
use crate::outline::OutlineModel;
use crate::tokenizer::{snap, Op, Vocab};
use crate::ufo::glyph_ops;

/// Glyphs never trained in any form; a completion stress test.
pub const EVAL_FULL_GLYPHS: &[&str] = &["R", "five"];
/// Glyphs whose singles train but whose pair is held out.
pub const EVAL_PAIR_GLYPHS: &[&str] = &["B", "g", "two", "hah-ar"];
/// Share of glyphs held out for validation loss.
pub const VAL_FRACTION: f64 = 0.05;
/// How often a sample of a bolded glyph is the pair.
pub const PAIR_PROB: f64 = 0.4;
/// How often a glyph name is dropped to `UNK`.
pub const UNK_PROB: f64 = 0.1;
/// The mark colour the lab treats as approved.
pub const GREEN: &str = "0.09,0.72,0.44,1";
/// The mark colour of an agent-conformed glyph.
pub const BLUE: &str = "0.27,0.44,1,1";

/// What a run is told.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Optimizer steps.
    pub steps: usize,
    /// Stop after this many minutes; 0 runs every step.
    pub minutes: f64,
    /// Sequences per step.
    pub batch: usize,
    /// Model width.
    pub dims: usize,
    /// Transformer blocks.
    pub layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Dropout inside the blocks.
    pub dropout: f64,
    /// Peak learning rate.
    pub lr: f64,
    /// Linear warmup steps before the cosine decay.
    pub warmup: usize,
    /// Validation and checkpoint every this many steps.
    pub ckpt_every: usize,
    /// Centre the delta tokens on the corpus mean, so predicting zero
    /// is the mean-shift baseline.
    pub recenter: bool,
    /// Which mark colours on the target master approve a glyph for
    /// training: `green`, `blue`, or `any`.
    pub colors: Vec<String>,
    /// Context length; 0 takes the corpus's longest sequence, at
    /// least 1024.
    pub max_len: usize,
    /// Seed for the split and the sampler.
    pub seed: u64,
    /// Train an adapter over `init` into this directory instead of a
    /// whole model. The base stays frozen.
    pub adapter_out: Option<PathBuf>,
    /// The adapter's rank.
    pub rank: usize,
    /// The adapter's alpha; the update is `alpha / rank` times `B·A`.
    pub alpha: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 2000,
            minutes: 0.0,
            batch: 24,
            dims: 384,
            layers: 6,
            heads: 8,
            dropout: 0.1,
            lr: 3e-4,
            warmup: 200,
            ckpt_every: 500,
            recenter: true,
            colors: vec!["green".into()],
            max_len: 0,
            seed: 7,
            adapter_out: None,
            rank: 8,
            alpha: 16.0,
        }
    }
}

/// What a run leaves behind, beside the model directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// The model directory.
    pub out: PathBuf,
    /// Steps taken; fewer than asked when the time ran out.
    pub steps: usize,
    /// The lowest validation loss seen, which is the checkpoint kept.
    pub best_val: f64,
    /// Loss at the first step, against `ln(vocab)` for a random model.
    pub init_loss: f64,
    /// Parameter count.
    pub params: usize,
    /// Wall time.
    pub seconds: f64,
    /// Vocabulary size.
    pub vocab: usize,
    /// Glyphs drawn in both masters and approved, which is what the
    /// pairs come from.
    pub pairs: usize,
    /// Glyphs in the training split.
    pub train_glyphs: usize,
    /// Held-out sequences.
    pub val_sequences: usize,
    /// The delta centre used.
    pub center: [i32; 2],
}

/// One master as the corpus reads it.
#[derive(Debug, Clone, Default)]
pub struct MasterGlyphs {
    /// Name to advance and outline, for glyphs with their own contours.
    pub glyphs: BTreeMap<String, (f64, Vec<Op>)>,
    /// Name to first Unicode value.
    pub unicodes: BTreeMap<String, u32>,
    /// Name to mark colour, as the glif carries it.
    pub marks: BTreeMap<String, String>,
}

/// Reads a UFO into what training needs.
pub fn load_master(path: &Path) -> Result<MasterGlyphs> {
    let font = norad::Font::load(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;
    let mut out = MasterGlyphs::default();
    for glyph in font.default_layer().iter() {
        let name = glyph.name().to_string();
        if let Some(ops) = glyph_ops(glyph) {
            out.glyphs.insert(name.clone(), (glyph.width, ops));
        }
        if let Some(cp) = glyph.codepoints.iter().next() {
            out.unicodes.insert(name.clone(), cp as u32);
        }
        if let Some(mark) = glyph
            .lib
            .get("public.markColor")
            .and_then(|v| v.as_string())
        {
            out.marks.insert(name, mark.to_string());
        }
    }
    Ok(out)
}

/// A small deterministic generator, so a run repeats on any machine.
/// SplitMix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// In `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// In `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

// ---- outline helpers, ported from the lab's tokenizer ----

/// The points an op carries.
fn points(op: &Op) -> Vec<(f64, f64)> {
    match op {
        Op::MoveTo(x, y) | Op::LineTo(x, y) => vec![(*x, *y)],
        Op::CurveTo(p) => p.to_vec(),
        Op::ClosePath => vec![],
    }
}

/// Where an op ends.
fn endpoint(op: &Op) -> Option<(f64, f64)> {
    points(op).last().copied()
}

/// Whether two outlines have the same commands in the same order.
pub fn compatible(a: &[Op], b: &[Op]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| std::mem::discriminant(x) == std::mem::discriminant(y))
}

/// Whether the target really differs from the base. A target copied
/// from the base and not yet drawn is debt, not signal.
pub fn has_boldening(base: &[Op], target: &[Op]) -> bool {
    base.iter().zip(target).any(|(a, b)| {
        points(a)
            .iter()
            .zip(points(b))
            .any(|(p, q)| (q.0 - p.0).abs() >= 8.0 || (q.1 - p.1).abs() >= 8.0)
    })
}

/// Splits an outline at each `ClosePath`.
pub fn split_contours(ops: &[Op]) -> Vec<Vec<Op>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    for op in ops {
        current.push(op.clone());
        if matches!(op, Op::ClosePath) {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Starts each closed contour at another of its points. Any rotation
/// draws the same shape, so this is free augmentation that stops the
/// model learning start points.
pub fn rotate_ops(ops: &[Op], rotations: &[usize]) -> Vec<Op> {
    let mut out = Vec::new();
    for (contour, &k) in split_contours(ops).iter().zip(rotations) {
        let closed = matches!(contour.last(), Some(Op::ClosePath));
        if !closed || contour.len() <= 2 {
            out.extend(contour.iter().cloned());
            continue;
        }
        let Op::MoveTo(mx, my) = contour[0] else {
            out.extend(contour.iter().cloned());
            continue;
        };
        let move_pt = (mx, my);
        let mut segs: Vec<Op> = contour[1..contour.len() - 1].to_vec();
        if segs.last().and_then(endpoint) != Some(move_pt) {
            // Make the implicit closing line explicit.
            segs.push(Op::LineTo(mx, my));
        }
        let k = k % segs.len();
        segs.rotate_left(k);
        let Some(new_move) = segs.last().and_then(endpoint) else {
            out.extend(contour.iter().cloned());
            continue;
        };
        if matches!(segs.last(), Some(Op::LineTo(..))) {
            // Re-implicitize the closing line at the new seam.
            segs.pop();
        }
        out.push(Op::MoveTo(new_move.0, new_move.1));
        out.extend(segs);
        out.push(Op::ClosePath);
    }
    out
}

/// Base-2 radical inverse: 1 → 1/2, 2 → 1/4, 3 → 3/4, 4 → 1/8. Every
/// value is exact in binary, and any prefix spreads evenly over
/// `[0, 1)`.
pub fn van_der_corput(mut n: usize) -> f64 {
    let mut q = 0.0;
    let mut denom = 1.0;
    while n > 0 {
        denom *= 2.0;
        q += (n & 1) as f64 / denom;
        n >>= 1;
    }
    q
}

/// Linear interpolation between compatible outlines, snapped to the
/// grid.
pub fn interpolate_ops(a: &[Op], b: &[Op], t: f64) -> Vec<Op> {
    let lerp = |p: (f64, f64), q: (f64, f64)| {
        (
            snap(p.0 + (q.0 - p.0) * t) as f64,
            snap(p.1 + (q.1 - p.1) * t) as f64,
        )
    };
    a.iter()
        .zip(b)
        .map(|(x, y)| match (x, y) {
            (Op::MoveTo(..), Op::MoveTo(qx, qy)) => {
                let (px, py) = points(x)[0];
                let (rx, ry) = lerp((px, py), (*qx, *qy));
                Op::MoveTo(rx, ry)
            }
            (Op::LineTo(..), Op::LineTo(qx, qy)) => {
                let (px, py) = points(x)[0];
                let (rx, ry) = lerp((px, py), (*qx, *qy));
                Op::LineTo(rx, ry)
            }
            (Op::CurveTo(p), Op::CurveTo(q)) => {
                Op::CurveTo([lerp(p[0], q[0]), lerp(p[1], q[1]), lerp(p[2], q[2])])
            }
            _ => x.clone(),
        })
        .collect()
}

// ---- encoders, matching the lab's sequence formats ----

fn header(v: &Vocab, name_tok: usize, uni: Option<u32>) -> Vec<u32> {
    let mut toks = vec![v.special("BOS").unwrap_or(1) as u32, name_tok as u32];
    if let Some(u) = v.unicode(uni) {
        toks.push(u as u32);
    }
    toks
}

/// `BOS NAME [U] W### ADV w  <ops>  EOS`.
pub fn encode_single(
    v: &Vocab,
    name_tok: usize,
    weight: u32,
    width: f64,
    ops: &[Op],
    uni: Option<u32>,
) -> Vec<u32> {
    let mut toks = header(v, name_tok, uni);
    toks.push(v.weight(weight) as u32);
    toks.push(v.special("ADV").unwrap_or(8) as u32);
    toks.push(v.coord(width) as u32);
    toks.extend(v.encode_ops(ops).into_iter().map(|t| t as u32));
    toks.push(v.special("EOS").unwrap_or(2) as u32);
    toks
}

/// `BOS NAME [U] PAIR ADV wR dwB  MOVE xR yR dxB dyB ... EOS`: both
/// masters in one stream, the target as offsets from the base after
/// every point. `center` is subtracted from every offset.
#[allow(clippy::too_many_arguments)]
pub fn encode_interleaved(
    v: &Vocab,
    name_tok: usize,
    width_r: f64,
    ops_r: &[Op],
    width_b: f64,
    ops_b: &[Op],
    center: (i32, i32),
    uni: Option<u32>,
) -> Vec<u32> {
    let (cx, cy) = (snap(center.0 as f64), snap(center.1 as f64));
    let mut toks = header(v, name_tok, uni);
    toks.push(v.special("PAIR").unwrap_or(13) as u32);
    toks.push(v.special("ADV").unwrap_or(8) as u32);
    toks.push(v.coord(width_r) as u32);
    toks.push(v.delta((snap(width_b) - snap(width_r) - cx) as f64) as u32);
    for (a, b) in ops_r.iter().zip(ops_b) {
        let name = match a {
            Op::ClosePath => {
                toks.push(v.special("CLOSE").unwrap_or(7) as u32);
                continue;
            }
            Op::MoveTo(..) => "MOVE",
            Op::LineTo(..) => "LINE",
            Op::CurveTo(..) => "CURVE",
        };
        toks.push(v.special(name).unwrap_or(4) as u32);
        for (p, q) in points(a).iter().zip(points(b)) {
            toks.push(v.coord(p.0) as u32);
            toks.push(v.coord(p.1) as u32);
            toks.push(v.delta((snap(q.0) - snap(p.0) - cx) as f64) as u32);
            toks.push(v.delta((snap(q.1) - snap(p.1) - cy) as f64) as u32);
        }
    }
    toks.push(v.special("EOS").unwrap_or(2) as u32);
    toks
}

/// The nearest weight token to an interpolation position.
fn weight_token(t: f64) -> u32 {
    let w = 400.0 + 300.0 * t;
    [400u32, 500, 600, 700]
        .into_iter()
        .min_by(|a, b| {
            ((*a as f64) - w)
                .abs()
                .partial_cmp(&((*b as f64) - w).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(400)
}

// ---- the corpus ----

/// Two masters, split and ready to sample.
pub struct Corpus {
    /// The token table.
    pub vocab: Vocab,
    base: MasterGlyphs,
    target: MasterGlyphs,
    /// Glyphs drawn in both.
    pub names: Vec<String>,
    /// The training split.
    pub train_names: Vec<String>,
    /// The validation split.
    pub val_names: Vec<String>,
    /// Approved, compatible, and really heavier in the target.
    pub bolded: BTreeSet<String>,
    /// Training glyphs whose pair is a sample.
    pub pair_names: BTreeSet<String>,
    /// The delta centre.
    pub center: (i32, i32),
    interp_i: usize,
}

impl Corpus {
    /// Builds the corpus. `vocab` pins a table from an earlier model
    /// when continuing from one.
    pub fn new(
        base: MasterGlyphs,
        target: MasterGlyphs,
        colors: &[String],
        recenter: bool,
        seed: u64,
        vocab: Option<Vocab>,
    ) -> Result<Self> {
        let names: Vec<String> = base
            .glyphs
            .keys()
            .filter(|n| target.glyphs.contains_key(*n))
            .cloned()
            .collect();
        if names.is_empty() {
            return Err(Error::Io {
                path: PathBuf::from("corpus"),
                source: std::io::Error::other(
                    "no glyph is drawn in both masters; nothing to train on",
                ),
            });
        }
        let unicodes: Vec<u32> = base.unicodes.values().copied().collect();
        let vocab = vocab.unwrap_or_else(|| Vocab::new(names.clone(), unicodes));

        let mut rng = Rng::new(seed);
        let mut pool: Vec<String> = names
            .iter()
            .filter(|n| !EVAL_FULL_GLYPHS.contains(&n.as_str()))
            .cloned()
            .collect();
        let val_count = ((pool.len() as f64 * VAL_FRACTION) as usize).max(8);
        // Fisher-Yates from the seed, then the first `val_count`.
        for i in (1..pool.len()).rev() {
            let j = rng.below(i + 1);
            pool.swap(i, j);
        }
        let mut val_names: Vec<String> = pool
            .iter()
            .take(val_count.min(pool.len()))
            .filter(|n| !EVAL_PAIR_GLYPHS.contains(&n.as_str()))
            .cloned()
            .collect();
        val_names.sort();
        let mut train_names: Vec<String> = names
            .iter()
            .filter(|n| !EVAL_FULL_GLYPHS.contains(&n.as_str()) && !val_names.contains(n))
            .cloned()
            .collect();
        train_names.sort();

        let any = colors.iter().any(|c| c == "any");
        let markers: Vec<&str> = colors
            .iter()
            .filter_map(|c| match c.as_str() {
                "green" => Some(GREEN),
                "blue" => Some(BLUE),
                _ => None,
            })
            .collect();
        let approved = |n: &str| {
            any || target
                .marks
                .get(n)
                .is_some_and(|m| markers.iter().any(|k| m.contains(k)))
        };
        let bolded: BTreeSet<String> = names
            .iter()
            .filter(|n| {
                let (_, a) = &base.glyphs[*n];
                let (_, b) = &target.glyphs[*n];
                approved(n) && compatible(a, b) && has_boldening(a, b)
            })
            .cloned()
            .collect();
        let pair_names: BTreeSet<String> = train_names
            .iter()
            .filter(|n| !EVAL_PAIR_GLYPHS.contains(&n.as_str()) && bolded.contains(*n))
            .cloned()
            .collect();

        let mut center = (0, 0);
        if recenter && !pair_names.is_empty() {
            let (mut n, mut sx, mut sy) = (0usize, 0.0, 0.0);
            for name in &pair_names {
                let (_, a) = &base.glyphs[name];
                let (_, b) = &target.glyphs[name];
                for (x, y) in a.iter().zip(b) {
                    for (p, q) in points(x).iter().zip(points(y)) {
                        sx += q.0 - p.0;
                        sy += q.1 - p.1;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                center = (snap(sx / n as f64), snap(sy / n as f64));
            }
        }
        Ok(Self {
            vocab,
            base,
            target,
            names,
            train_names,
            val_names,
            bolded,
            pair_names,
            center,
            interp_i: 0,
        })
    }

    fn name_token(&self, name: &str, rng: Option<&mut Rng>) -> usize {
        let unk = self.vocab.special("UNK").unwrap_or(14);
        let Ok(id) = self.vocab.name(name) else {
            return unk;
        };
        if let Some(rng) = rng {
            if rng.f64() < UNK_PROB {
                return unk;
            }
        }
        id
    }

    /// One training sequence, augmented.
    pub fn sample(&mut self, rng: &mut Rng) -> Vec<u32> {
        let name = self.train_names[rng.below(self.train_names.len())].clone();
        let as_pair = self.pair_names.contains(&name) && rng.f64() < PAIR_PROB;
        let (w_r, ops_r) = self.base.glyphs[&name].clone();
        let (w_b, ops_b) = self.target.glyphs[&name].clone();
        let rotations: Vec<usize> = split_contours(&ops_r)
            .iter()
            .map(|_| rng.below(64))
            .collect();
        let ops_r = rotate_ops(&ops_r, &rotations);
        let ops_b = rotate_ops(&ops_b, &rotations);
        let uni = self.base.unicodes.get(&name).copied();
        let name_tok = self.name_token(&name, Some(rng));
        if as_pair {
            encode_interleaved(
                &self.vocab,
                name_tok,
                w_r,
                &ops_r,
                w_b,
                &ops_b,
                self.center,
                uni,
            )
        } else if !self.bolded.contains(&name) {
            // Only the base is real: never teach the copied target
            // under a heavy label.
            encode_single(&self.vocab, name_tok, 400, w_r, &ops_r, uni)
        } else {
            self.interp_i += 1;
            let t = van_der_corput(self.interp_i);
            let ops_t = if compatible(&ops_r, &ops_b) {
                interpolate_ops(&ops_r, &ops_b, t)
            } else if t < 0.5 {
                ops_r.clone()
            } else {
                ops_b.clone()
            };
            let width_t = snap(w_r + (w_b - w_r) * t) as f64;
            encode_single(&self.vocab, name_tok, weight_token(t), width_t, &ops_t, uni)
        }
    }

    /// Fixed, unaugmented sequences over the held-out glyphs.
    pub fn val_sequences(&self) -> Vec<Vec<u32>> {
        let mut seqs = Vec::new();
        for name in &self.val_names {
            let (w_r, ops_r) = &self.base.glyphs[name];
            let (w_b, ops_b) = &self.target.glyphs[name];
            let uni = self.base.unicodes.get(name).copied();
            let tok = self.name_token(name, None);
            seqs.push(encode_single(&self.vocab, tok, 400, *w_r, ops_r, uni));
            if self.bolded.contains(name) {
                seqs.push(encode_single(&self.vocab, tok, 700, *w_b, ops_b, uni));
                seqs.push(encode_interleaved(
                    &self.vocab,
                    tok,
                    *w_r,
                    ops_r,
                    *w_b,
                    ops_b,
                    self.center,
                    uni,
                ));
            }
        }
        seqs
    }

    /// The longest pair sequence, plus room for the closing line a
    /// rotation can make explicit in each contour.
    pub fn max_len(&self) -> usize {
        let mut longest = 0;
        for name in &self.names {
            let (w_r, ops_r) = &self.base.glyphs[name];
            let (w_b, ops_b) = &self.target.glyphs[name];
            if compatible(ops_r, ops_b) {
                let tok = self.name_token(name, None);
                let len = encode_interleaved(
                    &self.vocab,
                    tok,
                    *w_r,
                    ops_r,
                    *w_b,
                    ops_b,
                    self.center,
                    None,
                )
                .len();
                longest = longest.max(len + 4 * split_contours(ops_r).len());
            }
        }
        longest
    }
}

// ---- the loop ----

/// Pads sequences to one width and makes the batch tensor.
fn pad_batch(seqs: &[Vec<u32>], pad: u32, max_len: usize, device: &Device) -> Result<Tensor> {
    let width = seqs
        .iter()
        .map(|s| s.len().min(max_len))
        .max()
        .unwrap_or(1)
        .max(2);
    let mut flat = Vec::with_capacity(seqs.len() * width);
    for s in seqs {
        let n = s.len().min(max_len);
        flat.extend_from_slice(&s[..n]);
        flat.extend(std::iter::repeat_n(pad, width - n));
    }
    Ok(Tensor::from_vec(flat, (seqs.len(), width), device)?)
}

/// Next-token cross entropy over every position that is not padding.
fn masked_loss(model: &OutlineModel, batch: &Tensor, pad: u32, train: bool) -> Result<Tensor> {
    let (b, l) = batch.dims2()?;
    let inputs = batch.narrow(1, 0, l - 1)?;
    let targets = batch.narrow(1, 1, l - 1)?;
    let logits = model.forward_batch(&inputs, train)?;
    let vocab = logits.dim(D::Minus1)?;
    let logits = logits.reshape((b * (l - 1), vocab))?;
    let targets = targets.reshape((b * (l - 1),))?;
    let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let picked = log_probs.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let mask = targets.ne(pad)?.to_dtype(DType::F32)?;
    let count = mask.sum_all()?;
    let loss = (picked.neg()? * &mask)?.sum_all()?;
    Ok((loss / count)?)
}

/// The learning rate at a step: linear warmup, then cosine to zero.
fn lr_at(step: usize, cfg: &TrainConfig) -> f64 {
    if step <= cfg.warmup {
        cfg.lr * step as f64 / cfg.warmup.max(1) as f64
    } else {
        let span = cfg.steps.saturating_sub(cfg.warmup).max(1) as f64;
        let t = (step - cfg.warmup) as f64 / span;
        cfg.lr * 0.5 * (1.0 + (std::f64::consts::PI * t.min(1.0)).cos())
    }
}

/// Today as `YYYY-MM-DD`, from the system clock.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    // Civil-from-days, Howard Hinnant's algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Trains a bolden model from `base` to `target` and writes it to
/// `out` as a model directory. `init` continues from a model already
/// there, pinning its vocabulary and shape. `progress` hears
/// `(step, steps, note)` every few steps.
#[allow(clippy::too_many_arguments)]
pub fn train(
    base: &Path,
    target: &Path,
    out: &Path,
    cfg: &TrainConfig,
    init: Option<&Path>,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<Report> {
    let started = Instant::now();
    let base_glyphs = load_master(base)?;
    let target_glyphs = load_master(target)?;

    // Continuing pins the earlier model's table and shape.
    let init_ckpt = init.map(Checkpoint::open).transpose()?;
    let pinned = init_ckpt
        .as_ref()
        .map(|c| Vocab::new(c.glyph_names.clone(), c.unicodes.clone()));
    let mut corpus = Corpus::new(
        base_glyphs,
        target_glyphs,
        &cfg.colors,
        cfg.recenter,
        cfg.seed,
        pinned,
    )?;
    if corpus.pair_names.is_empty() {
        return Err(Error::Io {
            path: target.to_path_buf(),
            source: std::io::Error::other(format!(
                "no approved pair to train on: no glyph in the target master is marked {} \
                 and drawn heavier than the base. Pass --colors any to train on every \
                 drawn glyph.",
                cfg.colors.join(",")
            )),
        });
    }
    let val = corpus.val_sequences();
    let vocab_size = corpus.vocab.len();
    let pad = corpus.vocab.special("PAD").unwrap_or(0) as u32;
    let max_len = match (&init_ckpt, cfg.max_len) {
        (Some(c), _) => c.config.max_len,
        (None, 0) => corpus.max_len().max(1024),
        (None, n) => n,
    };
    let model_cfg = ModelConfig {
        kind: ModelKind::Outline,
        dims: init_ckpt.as_ref().map_or(cfg.dims, |c| c.config.dims),
        layers: init_ckpt.as_ref().map_or(cfg.layers, |c| c.config.layers),
        heads: init_ckpt.as_ref().map_or(cfg.heads, |c| c.config.heads),
        vocab_size,
        max_len,
        delta_center: Some([corpus.center.0, corpus.center.1]),
        trim_close: false,
        extra: serde_json::Map::new(),
    };

    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let adapter = cfg.adapter_out.as_deref();
    let model = match (adapter, &init_ckpt) {
        (Some(_), None) => {
            return Err(Error::Io {
                path: out.to_path_buf(),
                source: std::io::Error::other("an adapter trains over a base: pass --init"),
            });
        }
        (Some(_), Some(base)) => {
            // The base is read as plain tensors, so nothing in it
            // learns; the adapter's matrices are the only variables.
            let base_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[base.weights_path()], DType::F32, &device)?
            };
            let mut model = OutlineModel::build(
                base_vb,
                &model_cfg,
                corpus.vocab.clone(),
                device.clone(),
                cfg.dropout,
            )?;
            let lora_vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            model.attach_lora(&lora_vb, cfg.rank, cfg.alpha, model_cfg.dims)?;
            model
        }
        (None, _) => {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = OutlineModel::build(
                vb,
                &model_cfg,
                corpus.vocab.clone(),
                device.clone(),
                cfg.dropout,
            )?;
            if let Some(c) = &init_ckpt {
                varmap.load(c.weights_path())?;
            }
            model
        }
    };
    let params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();

    // The directory written: the model's, or the adapter's.
    let out = adapter.unwrap_or(out);
    std::fs::create_dir_all(out).map_err(|source| Error::Io {
        path: out.to_path_buf(),
        source,
    })?;
    let weights_file = if adapter.is_some() {
        "adapter.safetensors"
    } else {
        "weights.safetensors"
    };
    if let Some(base) = init_ckpt.as_ref().filter(|_| adapter.is_some()) {
        write_json(
            &out.join("adapter.json"),
            &crate::adapter::AdapterConfig {
                rank: cfg.rank,
                alpha: cfg.alpha,
                base: base
                    .dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string(),
                targets: crate::adapter::TARGETS
                    .iter()
                    .map(|t| (*t).to_string())
                    .collect(),
                layers: model_cfg.layers,
            },
        )?;
    } else {
        write_vocab(out, &corpus.vocab)?;
        write_json(&out.join("config.json"), &model_cfg)?;
    }

    let mut opt = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: lr_at(1, cfg),
            ..Default::default()
        },
    )?;
    let mut rng = Rng::new(cfg.seed.wrapping_add(1));
    let deadline = (cfg.minutes > 0.0).then_some(cfg.minutes * 60.0);
    let mut best_val = f64::INFINITY;
    let mut init_loss = 0.0;
    let mut steps_done = 0;

    let val_loss = |model: &OutlineModel| -> Result<f64> {
        let mut total = 0.0;
        let mut count = 0;
        for chunk in val.chunks(cfg.batch.max(1)) {
            let batch = pad_batch(chunk, pad, max_len, &device)?;
            total += masked_loss(model, &batch, pad, false)?.to_scalar::<f32>()? as f64;
            count += 1;
        }
        Ok(if count == 0 {
            0.0
        } else {
            total / count as f64
        })
    };
    let checkpoint = |model: &OutlineModel, tag: &str, best: &mut f64| -> Result<()> {
        let v = val_loss(model)?;
        let mut note = format!("checkpoint {tag}: val loss {v:.4}");
        if v < *best {
            *best = v;
            varmap.save(out.join(weights_file))?;
            note.push_str(" (best, saved)");
        }
        eprintln!("{note}");
        Ok(())
    };

    // The starting point is a checkpoint too: an adapter begins as a
    // no-op and a continued model as itself, and a run that never
    // improves on that keeps it rather than writing something worse.
    if init_ckpt.is_some() {
        checkpoint(&model, "start", &mut best_val)?;
    }
    for step in 1..=cfg.steps {
        let seqs: Vec<Vec<u32>> = (0..cfg.batch).map(|_| corpus.sample(&mut rng)).collect();
        let batch = pad_batch(&seqs, pad, max_len, &device)?;
        let loss = masked_loss(&model, &batch, pad, true)?;
        opt.set_learning_rate(lr_at(step, cfg));
        opt.backward_step(&loss)?;
        let value = loss.to_scalar::<f32>()? as f64;
        steps_done = step;
        if step == 1 {
            init_loss = value;
            eprintln!(
                "init loss {value:.3}, a random model gives {:.3}",
                (vocab_size as f64).ln()
            );
        }
        if step == 1 || step % 10 == 0 {
            progress(step, cfg.steps, &format!("loss {value:.4}"));
        }
        if step % cfg.ckpt_every == 0 {
            checkpoint(&model, &format!("step {step}"), &mut best_val)?;
        }
        if deadline.is_some_and(|limit| started.elapsed().as_secs_f64() > limit) {
            eprintln!("time budget reached at step {step}");
            break;
        }
    }
    checkpoint(&model, "final", &mut best_val)?;

    let report = Report {
        out: out.to_path_buf(),
        steps: steps_done,
        best_val,
        init_loss,
        params,
        seconds: started.elapsed().as_secs_f64(),
        vocab: vocab_size,
        pairs: corpus.pair_names.len(),
        train_glyphs: corpus.train_names.len(),
        val_sequences: val.len(),
        center: [corpus.center.0, corpus.center.1],
    };
    let name = out
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();
    write_json(
        &out.join("manifest.json"),
        &serde_json::json!({
            "name": name,
            "task": "bolden",
            "kind": if adapter.is_some() { "adapter" } else { "model" },
            "trained": today(),
            "base": base,
            "target": target,
            "init": init,
            "steps": steps_done,
            "best_val_loss": best_val,
            "params": params,
            "pairs": report.pairs,
            "seconds": report.seconds,
            "notes": "Trained by font-ml train. Validation loss picked the checkpoint.",
        }),
    )?;
    Ok(report)
}

/// `vocab.txt` as `Checkpoint::open` reads it: names, then `#U` lines.
fn write_vocab(out: &Path, vocab: &Vocab) -> Result<()> {
    let mut text = String::new();
    for name in vocab.glyph_names() {
        text.push_str(name);
        text.push('\n');
    }
    for cp in vocab.unicodes() {
        text.push_str(&format!("#U {cp:04X}\n"));
    }
    let path = out.join("vocab.txt");
    std::fs::write(&path, text).map_err(|source| Error::Io { path, source })
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let text = serde_json::to_string_pretty(value).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;
    std::fs::write(path, text).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<Op> {
        vec![
            Op::MoveTo(0.0, 0.0),
            Op::LineTo(100.0, 0.0),
            Op::LineTo(100.0, 100.0),
            Op::LineTo(0.0, 100.0),
            Op::ClosePath,
        ]
    }

    #[test]
    fn rotation_keeps_the_shape_and_the_length() {
        let ops = square();
        for k in 0..6 {
            let r = rotate_ops(&ops, &[k]);
            assert_eq!(r.len(), ops.len(), "rotation {k}");
            assert!(matches!(r[0], Op::MoveTo(..)));
            assert!(matches!(r.last(), Some(Op::ClosePath)));
            let mut a: Vec<(i64, i64)> = ops
                .iter()
                .flat_map(points)
                .map(|p| (p.0 as i64, p.1 as i64))
                .collect();
            let mut b: Vec<(i64, i64)> = r
                .iter()
                .flat_map(points)
                .map(|p| (p.0 as i64, p.1 as i64))
                .collect();
            a.sort();
            b.sort();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn van_der_corput_is_the_dyadic_sequence() {
        assert_eq!(van_der_corput(1), 0.5);
        assert_eq!(van_der_corput(2), 0.25);
        assert_eq!(van_der_corput(3), 0.75);
        assert_eq!(van_der_corput(4), 0.125);
    }

    #[test]
    fn interpolation_is_exact_at_the_ends() {
        let a = square();
        let b: Vec<Op> = a
            .iter()
            .map(|o| match o {
                Op::MoveTo(x, y) => Op::MoveTo(x + 40.0, *y),
                Op::LineTo(x, y) => Op::LineTo(x + 40.0, *y),
                other => other.clone(),
            })
            .collect();
        assert_eq!(interpolate_ops(&a, &b, 0.0), a);
        assert_eq!(interpolate_ops(&a, &b, 1.0), b);
        assert!(has_boldening(&a, &b));
        assert!(compatible(&a, &b));
    }

    #[test]
    fn the_pair_encoding_has_four_tokens_per_point() {
        let v = Vocab::new(vec!["a".into()], vec![]);
        let a = square();
        let toks = encode_interleaved(&v, v.name("a").unwrap(), 200.0, &a, 240.0, &a, (0, 0), None);
        // BOS name PAIR ADV w dw, then MOVE + 4, three LINE + 4 each, CLOSE, EOS.
        assert_eq!(toks.len(), 6 + 5 + 3 * 5 + 1 + 1);
        assert_eq!(toks[0], v.special("BOS").unwrap() as u32);
        assert_eq!(*toks.last().unwrap(), v.special("EOS").unwrap() as u32);
    }

    #[test]
    fn the_schedule_warms_up_then_decays() {
        let cfg = TrainConfig {
            steps: 1000,
            warmup: 100,
            lr: 1.0,
            ..Default::default()
        };
        assert!(lr_at(50, &cfg) < lr_at(100, &cfg));
        assert!((lr_at(100, &cfg) - 1.0).abs() < 1e-9);
        assert!(lr_at(1000, &cfg) < 1e-6);
    }

    #[test]
    fn the_rng_repeats() {
        let mut a = Rng::new(3);
        let mut b = Rng::new(3);
        assert_eq!(a.next_u64(), b.next_u64());
        assert!(a.f64() < 1.0);
    }

    #[test]
    fn today_is_a_date() {
        let d = today();
        assert_eq!(d.len(), 10);
        assert!(d.starts_with("20"));
    }
}
