//! Reading glyph outlines out of UFO sources, and writing proposals
//! back in.
//!
//! Only drawn outlines are offered. A composite is a reference to
//! other glyphs, so predicting its outline would be answering the
//! wrong question: fix the base and the composite follows.
//!
//! A prediction goes back into the UFO as a *proposal*: a layer named
//! `com.runebender.proposal.<task>` holding the predicted glyphs. The
//! foreground is untouched. An editor that knows the convention
//! (Runebender does) shows the layer and lets the designer install or
//! discard it, one undo step per glyph. Any other UFO tool sees an
//! ordinary layer it can inspect or delete.

use crate::tokenizer::Op;

/// Every proposal layer starts with this.
pub const PROPOSAL_PREFIX: &str = "com.runebender.proposal.";

/// The layer a task's proposal goes in.
pub fn proposal_layer(task: &str) -> String {
    format!("{PROPOSAL_PREFIX}{task}")
}

/// Move a glyph's points by predicted offsets, in the order
/// [`glyph_ops`] read them, and return the moved contours.
///
/// Walks the same contours in the same rotation the reader used, so
/// offset *n* lands on the point it was predicted for. Point types
/// and smooth flags are left alone: this moves points and nothing
/// else. `center` is the checkpoint's `delta_center`, added to every
/// offset.
pub fn apply_deltas(
    glyph: &norad::Glyph,
    deltas: &[(i32, i32)],
    center: (i32, i32),
) -> Vec<norad::Contour> {
    let mut next = deltas.iter();
    let mut out = Vec::with_capacity(glyph.contours.len());
    for contour in &glyph.contours {
        let points = &contour.points;
        let start = points
            .iter()
            .position(|p| p.typ != norad::PointType::OffCurve)
            .unwrap_or(0);
        let n = points.len();
        let mut moved = points.clone();
        for step in 0..n {
            let i = (start + step) % n;
            let Some((dx, dy)) = next.next().copied() else {
                break;
            };
            moved[i].x += f64::from(dx + center.0);
            moved[i].y += f64::from(dy + center.1);
        }
        // The reader ends a closed contour by returning to its start,
        // so it yields one offset more than the contour has points.
        // Drop it, or every later contour is shifted by one point.
        next.next();
        out.push(norad::Contour::new(moved, contour.identifier().cloned()));
    }
    out
}

/// A predicted glyph as it goes into a proposal layer: the source
/// glyph's name and metadata, with the moved contours and the new
/// advance.
pub fn proposed_glyph(
    source: &norad::Glyph,
    contours: Vec<norad::Contour>,
    advance_delta: i32,
) -> norad::Glyph {
    let mut glyph = source.clone();
    glyph.contours = contours;
    glyph.width = source.width + f64::from(advance_delta);
    glyph
}

/// Put predicted glyphs into the task's proposal layer, replacing any
/// glyph of the same name already proposed. Nothing else in the font
/// changes. Returns the layer's name.
pub fn write_proposal(
    font: &mut norad::Font,
    task: &str,
    glyphs: impl IntoIterator<Item = norad::Glyph>,
) -> Result<String, norad::error::NamingError> {
    let name = proposal_layer(task);
    let layer = font.layers.get_or_create_layer(&name)?;
    for glyph in glyphs {
        layer.insert_glyph(glyph);
    }
    Ok(name)
}

/// The drawing commands of a glyph, or `None` if it has nothing to
/// draw or is built from components.
///
/// A closed contour ends by returning to its start, so the start point
/// is visited twice: once as the move and once as the end of the
/// closing segment. The command run therefore carries one point more
/// per contour than the contour stores, and anything mapping
/// per-point results back onto the source has to drop that duplicate
/// or every contour after the first is shifted by one point.
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
    fn deltas_land_on_the_points_they_were_predicted_for() {
        // Two contours. The reader yields one extra offset per contour
        // (the return to start), which must be skipped, or the second
        // contour's points get the wrong offsets.
        let mut g = norad::Glyph::new("x");
        g.contours.push(Contour::new(
            vec![
                point(0.0, 0.0, PointType::Line),
                point(10.0, 0.0, PointType::Line),
            ],
            None,
        ));
        g.contours.push(Contour::new(
            vec![
                point(20.0, 0.0, PointType::Line),
                point(30.0, 0.0, PointType::Line),
            ],
            None,
        ));
        let deltas = [(1, 0), (2, 0), (99, 99), (3, 0), (4, 0), (99, 99)];
        let moved = apply_deltas(&g, &deltas, (0, 0));
        let xs: Vec<f64> = moved
            .iter()
            .flat_map(|c| c.points.iter().map(|p| p.x))
            .collect();
        assert_eq!(xs, [1.0, 12.0, 23.0, 34.0]);
    }

    #[test]
    fn a_proposal_is_a_layer_beside_the_foreground() {
        let mut font = norad::Font::new();
        let mut g = norad::Glyph::new("a");
        g.width = 100.0;
        font.default_layer_mut().insert_glyph(g.clone());
        let proposed = proposed_glyph(&g, Vec::new(), 16);
        let layer = write_proposal(&mut font, "bolden", [proposed]).unwrap();
        assert_eq!(layer, "com.runebender.proposal.bolden");
        assert_eq!(font.default_layer().get_glyph("a").unwrap().width, 100.0);
        assert_eq!(
            font.layers
                .get(&layer)
                .unwrap()
                .get_glyph("a")
                .unwrap()
                .width,
            116.0
        );
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
