//! Boldening: predict one master's outline from another's.
//!
//! The method is structure-forced weight transfer. Every drawing
//! command and every source coordinate is *forced* from the input
//! glyph; the model is only allowed to fill the offset slots between
//! them, and its choice is restricted to the delta block of the
//! vocabulary. Two things follow.
//!
//! The output is point-compatible with the input by construction:
//! same contours, same points, same order, same types. Nothing is
//! added, removed or reordered, because there is no token position at
//! which the model could do so. That is the rule interpolation needs
//! and the rule a font's Bold master has to keep, and here it is not
//! a check performed afterwards but a property of the encoding.
//!
//! And the model answers a small question. Rather than draw a letter,
//! it says how far each point moves, choosing from a few hundred
//! offsets. That is a far easier thing to be right about.

use candle_core::{IndexOp, Tensor, D};

use crate::error::Result;
use crate::outline::OutlineModel;
use crate::tokenizer::{Op, Vocab};

/// A predicted master, and the input it came from.
#[derive(Debug, Clone)]
pub struct Bolden {
    /// The source outline, snapped to the grid.
    pub from: Vec<Op>,
    /// The predicted outline. Structurally identical to `from`.
    pub to: Vec<Op>,
    /// Predicted change in advance width.
    pub advance_delta: i32,
    /// Per-point offsets, in the order the points appear.
    pub deltas: Vec<(i32, i32)>,
}

impl Bolden {
    /// Whether the two outlines really are point-compatible. Always
    /// true by construction; a caller installing the result into a
    /// font can assert it rather than trust this module.
    pub fn is_compatible(&self) -> bool {
        self.from.len() == self.to.len()
            && self.from.iter().zip(&self.to).all(|(a, b)| {
                std::mem::discriminant(a) == std::mem::discriminant(b)
            })
    }
}

/// Predict the heavier master of `ops`.
///
/// `center` is the offset the model's deltas are measured from, and
/// comes from the checkpoint's `delta_center`.
pub fn bolden(
    model: &OutlineModel,
    glyph: &str,
    unicode: Option<u32>,
    advance: f64,
    ops: &[Op],
    center: (i32, i32),
) -> Result<Bolden> {
    let v = model.vocab();
    let mut tokens: Vec<u32> = Vec::new();

    // BOS, the glyph name, and the Unicode slot when the vocabulary
    // has one. A name the model never saw becomes UNK rather than an
    // error: the outline still carries most of the signal.
    tokens.push(v.special("BOS").unwrap() as u32);
    tokens.push(match v.name(glyph) {
        Ok(id) => id as u32,
        Err(_) => v.special("UNK").unwrap() as u32,
    });
    if let Some(uni) = v.unicode(unicode) {
        tokens.push(uni as u32);
    }
    tokens.push(v.special("PAIR").unwrap() as u32);
    tokens.push(v.special("ADV").unwrap() as u32);
    tokens.push(v.coord(advance) as u32);

    // The advance delta is the model's first real decision.
    let advance_delta = predict_delta(model, &mut tokens, v)?;

    let mut deltas = Vec::new();
    for op in ops {
        match op {
            Op::ClosePath => {
                tokens.push(v.special("CLOSE").unwrap() as u32);
            }
            _ => {
                let (name, points): (&str, Vec<(f64, f64)>) = match op {
                    Op::MoveTo(x, y) => ("MOVE", vec![(*x, *y)]),
                    Op::LineTo(x, y) => ("LINE", vec![(*x, *y)]),
                    Op::CurveTo(p) => ("CURVE", p.to_vec()),
                    Op::ClosePath => unreachable!(),
                };
                tokens.push(v.special(name).unwrap() as u32);
                for (x, y) in points {
                    // Forced: the source coordinate, not a prediction.
                    tokens.push(v.coord(x) as u32);
                    tokens.push(v.coord(y) as u32);
                    let dx = predict_delta(model, &mut tokens, v)?;
                    let dy = predict_delta(model, &mut tokens, v)?;
                    deltas.push((dx, dy));
                }
            }
        }
    }

    let from = snapped(v, ops);
    let to = apply(&from, &deltas, center);
    Ok(Bolden { from, to, advance_delta, deltas })
}

/// Ask the model for one offset and append it to the running sequence.
///
/// The choice is restricted to the delta block: an argmax over the
/// whole vocabulary could return a coordinate or a command, which
/// would put an absolute position where an offset belongs.
fn predict_delta(
    model: &OutlineModel,
    tokens: &mut Vec<u32>,
    v: &Vocab,
) -> Result<i32> {
    let logits = model.forward(tokens)?;
    let last = logits.i(logits.dim(0)? - 1)?;
    let id = argmax_delta(&last, v)?;
    tokens.push(id as u32);
    Ok(v.delta_value(id).unwrap_or(0))
}

/// Greedy over the delta block only.
///
/// Greedy on purpose: a designer reviewing a proposal should not have
/// to wonder whether running it again would have been better, and a
/// font wants the same answer twice.
fn argmax_delta(logits: &Tensor, v: &Vocab) -> Result<usize> {
    let base = v.delta_base();
    let count = v.delta_count();
    let slice = logits.narrow(D::Minus1, base, count)?;
    let best = slice.argmax(D::Minus1)?.to_scalar::<u32>()? as usize;
    Ok(base + best)
}

/// The input as the model saw it: coordinates on the grid.
fn snapped(v: &Vocab, ops: &[Op]) -> Vec<Op> {
    let c = |x: f64| v.coord_value(v.coord(x)).unwrap_or(0) as f64;
    ops.iter()
        .map(|op| match op {
            Op::MoveTo(x, y) => Op::MoveTo(c(*x), c(*y)),
            Op::LineTo(x, y) => Op::LineTo(c(*x), c(*y)),
            Op::CurveTo(p) => Op::CurveTo([
                (c(p[0].0), c(p[0].1)),
                (c(p[1].0), c(p[1].1)),
                (c(p[2].0), c(p[2].1)),
            ]),
            Op::ClosePath => Op::ClosePath,
        })
        .collect()
}

/// Add the offsets to the source points, in order.
fn apply(from: &[Op], deltas: &[(i32, i32)], center: (i32, i32)) -> Vec<Op> {
    let mut next = deltas.iter();
    let mut shift = |x: f64, y: f64| {
        let (dx, dy) = next.next().copied().unwrap_or((0, 0));
        (x + (dx + center.0) as f64, y + (dy + center.1) as f64)
    };
    from.iter()
        .map(|op| match op {
            Op::MoveTo(x, y) => {
                let (x, y) = shift(*x, *y);
                Op::MoveTo(x, y)
            }
            Op::LineTo(x, y) => {
                let (x, y) = shift(*x, *y);
                Op::LineTo(x, y)
            }
            Op::CurveTo(p) => Op::CurveTo([
                shift(p[0].0, p[0].1),
                shift(p[1].0, p[1].1),
                shift(p[2].0, p[2].1),
            ]),
            Op::ClosePath => Op::ClosePath,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applying_offsets_keeps_the_structure() {
        let from = vec![
            Op::MoveTo(0.0, 0.0),
            Op::LineTo(100.0, 0.0),
            Op::CurveTo([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
            Op::ClosePath,
        ];
        // One offset per point: 1 + 1 + 3 = 5.
        let deltas = vec![(10, 0), (10, 0), (1, 1), (1, 1), (1, 1)];
        let to = apply(&from, &deltas, (0, 0));
        let b = Bolden {
            from: from.clone(),
            to,
            advance_delta: 0,
            deltas,
        };
        assert!(b.is_compatible());
        assert_eq!(b.to[0], Op::MoveTo(10.0, 0.0));
        assert_eq!(b.to[3], Op::ClosePath);
    }

    #[test]
    fn a_short_delta_run_leaves_the_remaining_points_alone() {
        // A truncated prediction must not shift points by whatever
        // happens to be next in memory.
        let from = vec![Op::MoveTo(0.0, 0.0), Op::LineTo(10.0, 0.0)];
        let to = apply(&from, &[(5, 5)], (0, 0));
        assert_eq!(to[0], Op::MoveTo(5.0, 5.0));
        assert_eq!(to[1], Op::LineTo(10.0, 0.0));
    }

    #[test]
    fn the_delta_centre_is_added_to_every_point() {
        let from = vec![Op::MoveTo(0.0, 0.0)];
        let to = apply(&from, &[(0, 0)], (14, 0));
        assert_eq!(to[0], Op::MoveTo(14.0, 0.0));
    }
}
