//! Reading glyph outlines out of UFO sources.
//!
//! Only drawn outlines are offered. A composite is a reference to
//! other glyphs, so predicting its outline would be answering the
//! wrong question: fix the base and the composite follows.

use crate::tokenizer::Op;

/// The drawing commands of a glyph, or `None` if it has nothing to
/// draw or is built from components.
pub fn glyph_ops(glyph: &norad::Glyph) -> Option<Vec<Op>> {
    if !glyph.components.is_empty() || glyph.contours.is_empty() {
        return None;
    }
    let mut ops = Vec::new();
    for contour in &glyph.contours {
        ops.extend(contour_ops(contour)?);
    }
    (!ops.is_empty()).then_some(ops)
}

/// One contour, as move/line/curve/close.
///
/// UFO stores a closed contour as a ring of points with no explicit
/// start, so the run is rotated to begin at the first on-curve point,
/// which is what a pen replays.
fn contour_ops(contour: &norad::Contour) -> Option<Vec<Op>> {
    use norad::PointType;

    let points = &contour.points;
    if points.is_empty() {
        return None;
    }
    let start = points.iter().position(|p| p.typ != PointType::OffCurve)?;
    let n = points.len();
    let at = |i: usize| &points[(start + i) % n];

    let first = at(0);
    let mut ops = vec![Op::MoveTo(first.x, first.y)];
    let mut pending: Vec<(f64, f64)> = Vec::new();

    for i in 1..=n {
        let p = at(i % n);
        match p.typ {
            PointType::OffCurve => pending.push((p.x, p.y)),
            PointType::Line | PointType::Move => {
                ops.push(Op::LineTo(p.x, p.y));
                pending.clear();
            }
            PointType::Curve | PointType::QCurve => {
                if pending.len() == 2 {
                    ops.push(Op::CurveTo([pending[0], pending[1], (p.x, p.y)]));
                } else {
                    // A curve without two controls is not something
                    // this encoding can carry; treat it as a line
                    // rather than inventing control points.
                    ops.push(Op::LineTo(p.x, p.y));
                }
                pending.clear();
            }
        }
        if i == n {
            break;
        }
    }
    ops.push(Op::ClosePath);
    Some(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use norad::{Contour, ContourPoint, PointType};

    fn point(x: f64, y: f64, typ: PointType) -> ContourPoint {
        ContourPoint::new(x, y, typ, false, None, None)
    }

    #[test]
    fn a_rectangle_becomes_move_lines_close() {
        let c = Contour::new(
            vec![
                point(0.0, 0.0, PointType::Line),
                point(10.0, 0.0, PointType::Line),
                point(10.0, 10.0, PointType::Line),
                point(0.0, 10.0, PointType::Line),
            ],
            None,
        );
        let ops = contour_ops(&c).unwrap();
        assert_eq!(ops.len(), 6); // move + 3 lines + closing line + close
        assert_eq!(ops[0], Op::MoveTo(0.0, 0.0));
        assert_eq!(ops[5], Op::ClosePath);
    }

    #[test]
    fn a_contour_starting_off_curve_is_rotated_to_an_on_curve_point() {
        // UFO rings have no start; a pen needs one.
        let c = Contour::new(
            vec![
                point(5.0, 0.0, PointType::OffCurve),
                point(10.0, 5.0, PointType::OffCurve),
                point(10.0, 10.0, PointType::Curve),
                point(0.0, 0.0, PointType::Line),
            ],
            None,
        );
        let ops = contour_ops(&c).unwrap();
        assert_eq!(ops[0], Op::MoveTo(10.0, 10.0));
    }

    #[test]
    fn a_composite_has_no_outline_to_predict() {
        let mut g = norad::Glyph::new("aacute");
        g.components.push(norad::Component::new(
            norad::Name::new("a").unwrap(),
            Default::default(),
            None,
        ));
        assert!(glyph_ops(&g).is_none());
    }
}
