// Copyright 2026 the Runebender Authors
// SPDX-License-Identifier: Apache-2.0

//! Stem widths, measured by cutting the outline with a horizontal line.
//!
//! Average point error says whether a prediction landed near the
//! target. It does not say whether the result is usable, because the
//! same error spread evenly around a bowl is invisible and concentrated
//! on one side of a stem is a letter that reads wrong.
//!
//! What a reader notices is stem width. So measure it: cut the outline
//! at a height, and the widths of the inked spans are the stems.
//!
//! Nothing here knows about a particular family. The reference weight
//! comes from glyphs somebody already drew, which is the case that
//! matters: a master part-way finished, where the rest has to match
//! what is there.

use kurbo::{BezPath, Line, ParamCurve, PathSeg, Shape};

use crate::tokenizer::Op;

/// Turn drawing commands into a path that can be cut.
pub fn ops_to_path(ops: &[Op]) -> BezPath {
    let mut path = BezPath::new();
    for op in ops {
        match *op {
            Op::MoveTo(x, y) => path.move_to((x, y)),
            Op::LineTo(x, y) => path.line_to((x, y)),
            Op::CurveTo([a, b, c]) => path.curve_to(a, b, c),
            Op::ClosePath => path.close_path(),
        }
    }
    path
}

/// Widths of the inked spans where a horizontal line at `y` crosses.
///
/// Crossings are sorted, then each gap between neighbours is tested at
/// its midpoint. The gaps that are inside the outline are the ink.
///
/// Testing the midpoints rather than pairing crossings off matters on
/// real glyphs. Pairing assumes the even-odd rule and an even number
/// of crossings, and a line that grazes a curve's extreme or passes
/// through a node breaks both assumptions. Six of ten lowercase
/// letters measured nothing that way.
pub fn spans_at(path: &BezPath, y: f64) -> Vec<f64> {
    let bounds = path.bounding_box();
    if bounds.width() <= 0.0 || y <= bounds.y0 || y >= bounds.y1 {
        return Vec::new();
    }
    let cut = Line::new((bounds.x0 - 1.0, y), (bounds.x1 + 1.0, y));
    let mut crossings: Vec<f64> = Vec::new();
    for seg in path.segments() {
        for hit in seg.intersect_line(cut) {
            let p = match seg {
                PathSeg::Line(l) => l.eval(hit.segment_t),
                PathSeg::Quad(q) => q.eval(hit.segment_t),
                PathSeg::Cubic(c) => c.eval(hit.segment_t),
            };
            crossings.push(p.x);
        }
    }
    if crossings.len() < 2 {
        return Vec::new();
    }
    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut spans = Vec::new();
    for pair in crossings.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let width = b - a;
        if width <= 0.5 {
            continue;
        }
        if path.contains(kurbo::Point::new((a + b) / 2.0, y)) {
            spans.push(width);
        }
    }
    spans
}

/// A stem is a narrow member, so anything wider than this fraction of
/// the glyph is something else: a bowl cut off-centre, a crossbar, or
/// the whole letter where the line found no counter.
const WIDEST_STEM: f64 = 0.4;

/// The narrowest span at `y`, if it is narrow enough to be a stem.
///
/// `None` rather than a number when the line did not cut a stem. On an
/// `e` at half the x-height the cut lands on the bar and the span is
/// most of the letter; reporting that as a stem put a 140-unit error
/// into an average and made a working measurement look broken.
pub fn stem_at(path: &BezPath, y: f64) -> Option<f64> {
    let width = path.bounding_box().width();
    if width <= 0.0 {
        return None;
    }
    spans_at(path, y)
        .into_iter()
        .filter(|w| *w <= width * WIDEST_STEM)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    })
}

/// The weight a set of already-drawn glyphs is carrying.
///
/// The median across them, so one odd letter cannot set the target for
/// everything that follows.
pub fn reference_stem(paths: &[BezPath], y: f64) -> Option<f64> {
    median(paths.iter().filter_map(|p| stem_at(p, y)).collect())
}

/// How far a prediction's stem is from the weight it should carry.
///
/// `None` when either outline cannot be measured at that height, which
/// is honest: a glyph with no stem there has nothing to compare.
pub fn stem_error(predicted: &BezPath, target_stem: f64, y: f64) -> Option<f64> {
    stem_at(predicted, y).map(|s| s - target_stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        let mut p = BezPath::new();
        p.move_to((x0, y0));
        p.line_to((x1, y0));
        p.line_to((x1, y1));
        p.line_to((x0, y1));
        p.close_path();
        p
    }

    /// Wound the other way, the way a counter is drawn against its
    /// outer contour. Inside is decided by winding, so a counter that
    /// runs the same direction as its outer contour is not a counter.
    fn hole(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        let mut p = BezPath::new();
        p.move_to((x0, y0));
        p.line_to((x0, y1));
        p.line_to((x1, y1));
        p.line_to((x1, y0));
        p.close_path();
        p
    }

    /// A stem is measured inside a glyph. A shape that is nothing but
    /// one solid block has no stem, which `narrowness_tests` covers.
    #[test]
    fn a_stem_measures_its_width() {
        let mut p = rect(0.0, 0.0, 100.0, 500.0);
        p.extend(rect(400.0, 0.0, 500.0, 500.0));
        assert_eq!(spans_at(&p, 250.0), vec![100.0, 100.0]);
        assert_eq!(stem_at(&p, 250.0), Some(100.0));
    }

    /// Two stems, as in an `n` or an `H`: both spans come back, and the
    /// narrower one is what `stem_at` reports.
    #[test]
    fn two_stems_are_two_spans() {
        let mut p = rect(0.0, 0.0, 90.0, 500.0);
        p.extend(rect(200.0, 0.0, 300.0, 500.0));
        let spans = spans_at(&p, 250.0);
        assert_eq!(spans.len(), 2);
        assert_eq!(stem_at(&p, 250.0), Some(90.0));
    }

    /// A counter must not be counted as ink. An `o` cut across the
    /// middle is two stems, not one span the width of the letter.
    #[test]
    fn a_counter_is_not_ink() {
        let mut p = rect(0.0, 0.0, 200.0, 500.0);
        p.extend(hole(40.0, 40.0, 160.0, 460.0));
        let spans = spans_at(&p, 250.0);
        assert_eq!(spans.len(), 2, "left wall and right wall: {spans:?}");
        assert!((spans[0] - 40.0).abs() < 1e-6, "{spans:?}");
    }

    #[test]
    fn a_line_outside_the_glyph_measures_nothing() {
        let mut p = rect(0.0, 0.0, 100.0, 500.0);
        p.extend(rect(400.0, 0.0, 500.0, 500.0));
        assert!(spans_at(&p, 900.0).is_empty());
        assert!(spans_at(&p, -10.0).is_empty());
    }

    /// The reference is a median, so one heavy glyph among light ones
    /// does not drag the target it sets for everything else.
    #[test]
    fn the_reference_is_a_median() {
        let pair = |w: f64| {
            let mut p = rect(0.0, 0.0, w, 500.0);
            p.extend(rect(900.0, 0.0, 900.0 + w, 500.0));
            p
        };
        let paths = vec![pair(100.0), pair(100.0), pair(300.0)];
        assert_eq!(reference_stem(&paths, 250.0), Some(100.0));
    }

    #[test]
    fn stem_error_is_signed() {
        let pair = |w: f64| {
            let mut p = rect(0.0, 0.0, w, 500.0);
            p.extend(rect(900.0, 0.0, 900.0 + w, 500.0));
            p
        };
        assert_eq!(stem_error(&pair(80.0), 100.0, 250.0), Some(-20.0));
        assert_eq!(stem_error(&pair(130.0), 100.0, 250.0), Some(30.0));
    }
}

/// The strength that would land a prediction on the weight you want.
///
/// The model moves points; multiplying those offsets by `s` scales how
/// far they move, and the stem width that results is very nearly
/// linear in `s`. So rather than search, solve:
///
/// ```text
/// stem(s) ~= stem_regular + s * (stem_at_1 - stem_regular)
/// ```
///
/// This is what turns a model that is right about shape and wrong
/// about weight into a usable one. It needs a target, and the honest
/// source for that is glyphs somebody already drew in the heavier
/// master: match what is there rather than guess.
///
/// `None` when either outline cannot be measured, or when the model
/// barely moved the stem at all, where dividing by that difference
/// would turn noise into a huge multiplier.
pub fn fit_strength(
    regular: &BezPath,
    predicted: &BezPath,
    target_stem: f64,
    y: f64,
) -> Option<f64> {
    let from = stem_at(regular, y)?;
    let at_one = stem_at(predicted, y)?;
    let moved = at_one - from;
    if moved.abs() < 1.0 {
        return None;
    }
    Some((target_stem - from) / moved)
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    /// Two stems of the given width, far enough apart that each is a
    /// small part of the glyph, the way a stem is in a letter.
    fn bar(width: f64) -> BezPath {
        let mut p = BezPath::new();
        for x in [0.0, 900.0] {
            p.move_to((x, 0.0));
            p.line_to((x + width, 0.0));
            p.line_to((x + width, 500.0));
            p.line_to((x, 500.0));
            p.close_path();
        }
        p
    }

    /// The model moved the stem half as far as it should have, so the
    /// strength that lands it is 2.
    #[test]
    fn an_under_bold_prediction_wants_more_strength() {
        let regular = bar(96.0);
        let predicted = bar(144.0);
        let s = fit_strength(&regular, &predicted, 192.0, 250.0).expect("measurable");
        assert!((s - 2.0).abs() < 1e-6, "{s}");
    }

    #[test]
    fn a_correct_prediction_wants_strength_one() {
        let s = fit_strength(&bar(96.0), &bar(192.0), 192.0, 250.0).expect("measurable");
        assert!((s - 1.0).abs() < 1e-6, "{s}");
    }

    /// A model that did not move the stem gives no information about
    /// how far to scale, and scaling nothing by a large number is
    /// still nothing. Better to say so than to return a wild figure.
    #[test]
    fn a_model_that_moved_nothing_cannot_be_fitted() {
        assert!(fit_strength(&bar(96.0), &bar(96.0), 192.0, 250.0).is_none());
    }

    #[test]
    fn an_unmeasurable_height_cannot_be_fitted() {
        assert!(fit_strength(&bar(96.0), &bar(144.0), 192.0, 900.0).is_none());
    }
}

#[cfg(test)]
mod narrowness_tests {
    use super::*;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        let mut p = BezPath::new();
        p.move_to((x0, y0));
        p.line_to((x1, y0));
        p.line_to((x1, y1));
        p.line_to((x0, y1));
        p.close_path();
        p
    }

    /// A solid block is not a stem, however narrow the glyph is.
    /// Without this an `e` cut across its bar reports the whole letter.
    #[test]
    fn a_solid_span_is_not_a_stem() {
        let p = rect(0.0, 0.0, 400.0, 500.0);
        assert_eq!(spans_at(&p, 250.0), vec![400.0]);
        assert_eq!(stem_at(&p, 250.0), None, "the whole letter is not a stem");
    }

    #[test]
    fn a_narrow_member_is_a_stem() {
        let mut p = rect(0.0, 0.0, 90.0, 500.0);
        p.extend(rect(310.0, 0.0, 400.0, 500.0));
        assert_eq!(stem_at(&p, 250.0), Some(90.0));
    }
}

/// How much weight a pair of masters adds, learned from glyphs drawn
/// in both.
///
/// This is the "draw n, o, H, O and let the reference carry the rest"
/// workflow. Rather than a single target every glyph is pushed to, the
/// reference pairs say how far the weight moved, and each glyph is
/// asked to move by the same amount from wherever it already sits.
///
/// A constant delta rather than a ratio because that is how a
/// systematic family is drawn. Virtua Grotesk's notes put the growth
/// in quanta: verticals +96, bars +72, curve horizontals +48. Its cap
/// and lowercase stems both add 96, from 104 and 96, which one target
/// cannot express and one delta can.
///
/// The median across the pairs, so one odd reference cannot set the
/// weight for the whole master.
pub fn reference_delta(pairs: &[(BezPath, BezPath)], y: f64) -> Option<f64> {
    let deltas: Vec<f64> = pairs
        .iter()
        .filter_map(|(light, heavy)| Some(stem_at(heavy, y)? - stem_at(light, y)?))
        .collect();
    median(deltas)
}

/// The weight a glyph should carry: what it has now, plus what the
/// reference pairs added.
pub fn target_from_delta(regular: &BezPath, delta: f64, y: f64) -> Option<f64> {
    Some(stem_at(regular, y)? + delta)
}

#[cfg(test)]
mod reference_tests {
    use super::*;

    fn bar(width: f64) -> BezPath {
        let mut p = BezPath::new();
        for x in [0.0, 900.0] {
            p.move_to((x, 0.0));
            p.line_to((x + width, 0.0));
            p.line_to((x + width, 500.0));
            p.line_to((x, 500.0));
            p.close_path();
        }
        p
    }

    /// Two references that add the same amount from different starting
    /// weights: the delta is what they agree on, and it is what a
    /// third glyph should be asked to add.
    #[test]
    fn the_reference_teaches_a_delta_not_a_target() {
        let pairs = vec![
            (bar(96.0), bar(192.0)),  // lowercase: +96
            (bar(104.0), bar(200.0)), // caps: also +96
        ];
        let delta = reference_delta(&pairs, 250.0).expect("measurable");
        assert_eq!(delta, 96.0);
        // A glyph starting somewhere else still adds the same.
        let target = target_from_delta(&bar(100.0), delta, 250.0).expect("measurable");
        assert_eq!(target, 196.0);
    }

    /// The failure a single target would cause: pushing the caps to
    /// the lowercase weight, losing the 8 units between them.
    #[test]
    fn a_delta_keeps_weights_that_differ_apart() {
        let pairs = vec![(bar(96.0), bar(192.0)), (bar(104.0), bar(200.0))];
        let delta = reference_delta(&pairs, 250.0).expect("measurable");
        let lower = target_from_delta(&bar(96.0), delta, 250.0).unwrap();
        let caps = target_from_delta(&bar(104.0), delta, 250.0).unwrap();
        assert_eq!(caps - lower, 8.0, "the two weights stay apart");
    }

    #[test]
    fn a_reference_that_cannot_be_measured_gives_nothing() {
        assert!(reference_delta(&[], 250.0).is_none());
    }
}
