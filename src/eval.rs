//! Scoring a prediction against a master somebody drew.
//!
//! A prediction that moves many points is not thereby a good one. The
//! question is whether it lands nearer the real thing than the dumb
//! answer does, and the dumb answer here is the mean offset: shift
//! every point by the average the corpus moves. A model that cannot
//! beat that has not learned anything worth running.

use crate::tokenizer::Op;

/// How one glyph's prediction did.
#[derive(Debug, Clone)]
pub struct Score {
    pub glyph: String,
    pub points: usize,
    /// Mean absolute error of the prediction, in font units.
    pub model: f64,
    /// Mean absolute error of shifting every point by the mean offset.
    pub baseline: f64,
}

impl Score {
    /// How much better than the baseline, as a fraction. Negative
    /// means the baseline won.
    pub fn improvement(&self) -> f64 {
        if self.baseline == 0.0 {
            return 0.0;
        }
        (self.baseline - self.model) / self.baseline
    }
}

/// Every point of an outline, in order.
pub fn points(ops: &[Op]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for op in ops {
        match op {
            Op::MoveTo(x, y) | Op::LineTo(x, y) => out.push((*x, *y)),
            Op::CurveTo(p) => out.extend_from_slice(p),
            Op::ClosePath => {}
        }
    }
    out
}

/// Mean absolute error between two point lists, per coordinate.
pub fn mean_abs_error(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return f64::NAN;
    }
    let total: f64 = a
        .iter()
        .zip(b)
        .map(|(p, q)| (p.0 - q.0).abs() + (p.1 - q.1).abs())
        .sum();
    total / (a.len() as f64 * 2.0)
}

/// Stem widths at `y`, for the prediction and for the real thing.
///
/// Reported next to the point error because they answer different
/// questions. Point error says how close the prediction landed. Stem
/// width says whether the result carries the right weight, which is
/// what a reader sees and what the point error cannot tell you: the
/// same error spread around a bowl is invisible, and concentrated on
/// one side of a stem it is a letter that reads wrong.
pub fn stem_comparison(predicted: &[Op], actual: &[Op], y: f64) -> Option<(f64, f64)> {
    let p = crate::stems::stem_at(&crate::stems::ops_to_path(predicted), y)?;
    let a = crate::stems::stem_at(&crate::stems::ops_to_path(actual), y)?;
    Some((p, a))
}

/// Score a prediction against the real thing.
///
/// `mean_delta` is the baseline shift, normally the corpus mean the
/// model was centred on.
pub fn score(
    glyph: &str,
    predicted: &[Op],
    actual: &[Op],
    regular: &[Op],
    mean_delta: (f64, f64),
) -> Score {
    let pred = points(predicted);
    let real = points(actual);
    let reg = points(regular);
    let baseline: Vec<(f64, f64)> = reg
        .iter()
        .map(|(x, y)| (x + mean_delta.0, y + mean_delta.1))
        .collect();
    Score {
        glyph: glyph.to_string(),
        points: real.len(),
        model: mean_abs_error(&pred, &real),
        baseline: mean_abs_error(&baseline, &real),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_perfect_prediction_scores_zero() {
        let ops = vec![Op::MoveTo(0.0, 0.0), Op::LineTo(10.0, 0.0)];
        let s = score("x", &ops, &ops, &ops, (0.0, 0.0));
        assert_eq!(s.model, 0.0);
    }

    #[test]
    fn improvement_is_negative_when_the_baseline_wins() {
        let s = Score {
            glyph: "x".into(),
            points: 4,
            model: 20.0,
            baseline: 10.0,
        };
        assert!(s.improvement() < 0.0);
    }

    #[test]
    fn mismatched_lengths_do_not_silently_compare() {
        let a = [(0.0, 0.0)];
        let b = [(0.0, 0.0), (1.0, 1.0)];
        assert!(mean_abs_error(&a, &b).is_nan());
    }
}

#[cfg(test)]
mod stem_tests {
    use super::*;

    /// Two stems, so each is a narrow part of the glyph rather than
    /// the whole of it.
    fn bar(width: f64) -> Vec<Op> {
        let mut ops = Vec::new();
        for x in [0.0f64, 900.0] {
            ops.extend([
                Op::MoveTo(x, 0.0),
                Op::LineTo(x + width, 0.0),
                Op::LineTo(x + width, 500.0),
                Op::LineTo(x, 500.0),
                Op::ClosePath,
            ]);
        }
        ops
    }

    /// The case that makes stem width worth reporting: a prediction
    /// can sit close on average and still carry the wrong weight.
    #[test]
    fn stems_are_compared_at_a_height() {
        let predicted = bar(90.0);
        let actual = bar(120.0);
        let (p, a) = stem_comparison(&predicted, &actual, 250.0).expect("both measurable");
        assert_eq!((p, a), (90.0, 120.0));
    }

    #[test]
    fn an_unmeasurable_height_reports_nothing() {
        assert!(stem_comparison(&bar(90.0), &bar(120.0), 900.0).is_none());
    }
}
