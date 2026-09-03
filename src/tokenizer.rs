//! Outlines to token sequences and back.
//!
//! A port of the training lab's Python tokenizer. The two must agree
//! exactly: an id that means `LINE` here and something else there
//! produces confident nonsense rather than an error.
//!
//! The scheme rests on a grid. On a 2-unit grid every coordinate in
//! the usable range collapses to one of a few hundred values, so an
//! outline becomes a short sequence over a small vocabulary. Ids are
//! laid out in fixed blocks: specials, glyph names, coordinates,
//! deltas, then Unicode values last, so that a vocabulary saved before
//! Unicode tokens existed still loads with the same ids.

use crate::error::{Error, Result};

/// Coordinate range the grid covers, and its step.
pub const COORD_MIN: i32 = -512;
pub const COORD_MAX: i32 = 1280;
pub const GRID: i32 = 2;
/// Number of coordinate tokens: 897 on the default grid.
pub const N_COORDS: usize = ((COORD_MAX - COORD_MIN) / GRID + 1) as usize;

/// Range of Bold-minus-Regular offsets the delta tokens cover.
pub const DELTA_MIN: i32 = -384;
pub const DELTA_MAX: i32 = 384;
pub const N_DELTAS: usize = ((DELTA_MAX - DELTA_MIN) / GRID + 1) as usize;

/// Tokens that are neither names nor numbers. Order is part of the
/// format: these occupy ids 0..15.
pub const SPECIALS: &[&str] = &[
    "PAD", "BOS", "EOS", "SEP", "MOVE", "LINE", "CURVE", "CLOSE", "ADV", "W400", "W500", "W600",
    "W700", "PAIR", "UNK",
];

/// A drawing command, as the model sees it.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    MoveTo(f64, f64),
    LineTo(f64, f64),
    /// Cubic: two control points then the on-curve point.
    CurveTo([(f64, f64); 3]),
    ClosePath,
}

/// Snap a coordinate to the grid and clamp it to the covered range.
pub fn snap(v: f64) -> i32 {
    let q = (v / GRID as f64).round() as i32 * GRID;
    q.clamp(COORD_MIN, COORD_MAX)
}

/// The token table for one model.
#[derive(Debug, Clone)]
pub struct Vocab {
    tokens: Vec<String>,
    glyph_names: Vec<String>,
    unicodes: Vec<u32>,
    coord_base: usize,
    delta_base: usize,
    unicode_base: Option<usize>,
}

impl Vocab {
    /// Build the table from the glyph names and Unicode values a
    /// `vocab.txt` carries. Both must already be sorted the way the
    /// lab sorts them, which `Checkpoint::open` preserves.
    pub fn new(mut glyph_names: Vec<String>, mut unicodes: Vec<u32>) -> Self {
        glyph_names.sort();
        unicodes.sort_unstable();
        unicodes.dedup();

        let mut tokens: Vec<String> = SPECIALS.iter().map(|s| s.to_string()).collect();
        tokens.extend(glyph_names.iter().map(|n| format!("N_{n}")));

        let coord_base = tokens.len();
        let mut v = COORD_MIN;
        while v <= COORD_MAX {
            tokens.push(format!("C{v}"));
            v += GRID;
        }

        let delta_base = tokens.len();
        let mut d = DELTA_MIN;
        while d <= DELTA_MAX {
            tokens.push(format!("D{d}"));
            d += GRID;
        }

        let unicode_base = if unicodes.is_empty() {
            None
        } else {
            let base = tokens.len();
            tokens.push("U_NONE".to_string());
            tokens.extend(unicodes.iter().map(|cp| format!("U_{cp:04X}")));
            Some(base)
        };

        Self {
            tokens,
            glyph_names,
            unicodes,
            coord_base,
            delta_base,
            unicode_base,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// The glyph names, in vocabulary order.
    pub fn glyph_names(&self) -> &[String] {
        &self.glyph_names
    }

    /// The Unicode values, sorted.
    pub fn unicodes(&self) -> &[u32] {
        &self.unicodes
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn token(&self, id: usize) -> Option<&str> {
        self.tokens.get(id).map(String::as_str)
    }

    /// Id of a named token, such as `BOS` or `CURVE`.
    pub fn special(&self, name: &str) -> Option<usize> {
        SPECIALS.iter().position(|s| *s == name)
    }

    /// Id of a glyph-name token.
    pub fn name(&self, glyph: &str) -> Result<usize> {
        self.glyph_names
            .binary_search(&glyph.to_string())
            .map(|i| SPECIALS.len() + i)
            .map_err(|_| Error::UnknownGlyph(glyph.to_string()))
    }

    /// Id of a coordinate, snapped to the grid.
    pub fn coord(&self, v: f64) -> usize {
        self.coord_base + ((snap(v) - COORD_MIN) / GRID) as usize
    }

    /// The coordinate a token stands for, or `None` if it is not one.
    pub fn coord_value(&self, id: usize) -> Option<i32> {
        (id >= self.coord_base && id < self.delta_base)
            .then(|| COORD_MIN + ((id - self.coord_base) as i32) * GRID)
    }

    /// First id in the delta block, for constrained sampling.
    pub fn delta_base(&self) -> usize {
        self.delta_base
    }

    /// How many delta tokens there are.
    pub fn delta_count(&self) -> usize {
        N_DELTAS
    }

    /// Id of a delta, snapped and clamped.
    pub fn delta(&self, v: f64) -> usize {
        let q = ((v / GRID as f64).round() as i32 * GRID).clamp(DELTA_MIN, DELTA_MAX);
        self.delta_base + ((q - DELTA_MIN) / GRID) as usize
    }

    /// The delta a token stands for, or `None`.
    pub fn delta_value(&self, id: usize) -> Option<i32> {
        let end = self.unicode_base.unwrap_or(self.tokens.len());
        (id >= self.delta_base && id < end)
            .then(|| DELTA_MIN + ((id - self.delta_base) as i32) * GRID)
    }

    /// Conditioning token for a Unicode value; `U_NONE` when the glyph
    /// has none or the value is not in the vocabulary.
    pub fn unicode(&self, cp: Option<u32>) -> Option<usize> {
        let base = self.unicode_base?;
        let Some(cp) = cp else { return Some(base) };
        Some(match self.unicodes.binary_search(&cp) {
            Ok(i) => base + 1 + i,
            Err(_) => base,
        })
    }

    /// Weight conditioning token, falling back to `UNK` off the ladder.
    pub fn weight(&self, weight: u32) -> usize {
        let name = match weight {
            400 => "W400",
            500 => "W500",
            600 => "W600",
            700 => "W700",
            _ => "UNK",
        };
        self.special(name).expect("specials are always present")
    }

    /// Encode drawing commands. Coordinates are snapped to the grid.
    pub fn encode_ops(&self, ops: &[Op]) -> Vec<usize> {
        let mut out = Vec::new();
        for op in ops {
            match op {
                Op::MoveTo(x, y) => {
                    out.push(self.special("MOVE").unwrap());
                    out.push(self.coord(*x));
                    out.push(self.coord(*y));
                }
                Op::LineTo(x, y) => {
                    out.push(self.special("LINE").unwrap());
                    out.push(self.coord(*x));
                    out.push(self.coord(*y));
                }
                Op::CurveTo(points) => {
                    out.push(self.special("CURVE").unwrap());
                    for (x, y) in points {
                        out.push(self.coord(*x));
                        out.push(self.coord(*y));
                    }
                }
                Op::ClosePath => out.push(self.special("CLOSE").unwrap()),
            }
        }
        out
    }

    /// Read drawing commands back out of a token sequence.
    ///
    /// Tolerant by design: a model can emit a command and then stop, or
    /// emit something that is not a coordinate where one belongs. Such
    /// a run ends the decode rather than failing it, so a caller gets
    /// the part of the outline that was well formed.
    pub fn decode_ops(&self, ids: &[usize]) -> Vec<Op> {
        let mut ops = Vec::new();
        let mut i = 0;
        let coord_at = |i: usize| -> Option<f64> {
            ids.get(i)
                .and_then(|id| self.coord_value(*id))
                .map(|v| v as f64)
        };
        while i < ids.len() {
            let Some(token) = self.token(ids[i]) else {
                break;
            };
            match token {
                "MOVE" | "LINE" => {
                    let (Some(x), Some(y)) = (coord_at(i + 1), coord_at(i + 2)) else {
                        break;
                    };
                    ops.push(if token == "MOVE" {
                        Op::MoveTo(x, y)
                    } else {
                        Op::LineTo(x, y)
                    });
                    i += 3;
                }
                "CURVE" => {
                    let mut pts = [(0.0, 0.0); 3];
                    let mut ok = true;
                    for (n, slot) in pts.iter_mut().enumerate() {
                        match (coord_at(i + 1 + n * 2), coord_at(i + 2 + n * 2)) {
                            (Some(x), Some(y)) => *slot = (x, y),
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        break;
                    }
                    ops.push(Op::CurveTo(pts));
                    i += 7;
                }
                "CLOSE" => {
                    ops.push(Op::ClosePath);
                    i += 1;
                }
                "EOS" => break,
                // Header tokens before the drawing, or padding after it.
                _ => i += 1,
            }
        }
        ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec!["A".into(), "B".into()], vec![0x41, 0x42])
    }

    #[test]
    fn the_id_layout_matches_the_training_lab() {
        let v = vocab();
        // Specials first, then names, then coordinates, then deltas,
        // then Unicode last. Anything else silently mistrains.
        assert_eq!(v.special("PAD"), Some(0));
        assert_eq!(v.special("BOS"), Some(1));
        assert_eq!(v.name("A").unwrap(), SPECIALS.len());
        assert_eq!(v.name("B").unwrap(), SPECIALS.len() + 1);
        assert_eq!(v.coord(COORD_MIN as f64), SPECIALS.len() + 2);
        assert_eq!(N_COORDS, 897);
        assert_eq!(N_DELTAS, 385);
        // specials + 2 names + coords + deltas + U_NONE + 2 unicodes
        assert_eq!(v.len(), 15 + 2 + 897 + 385 + 1 + 2);
    }

    #[test]
    fn coordinates_round_trip_through_the_grid() {
        let v = vocab();
        for value in [-512.0, -100.0, 0.0, 1.0, 511.0, 1280.0] {
            let id = v.coord(value);
            assert_eq!(v.coord_value(id), Some(snap(value)));
        }
    }

    #[test]
    fn coordinates_snap_to_even_units() {
        assert_eq!(snap(0.4), 0);
        assert_eq!(snap(1.0), 2); // .round() breaks ties upward
        assert_eq!(snap(101.0), 102);
        assert_eq!(snap(100.9), 100);
    }

    #[test]
    fn coordinates_outside_the_range_clamp() {
        assert_eq!(snap(-9000.0), COORD_MIN);
        assert_eq!(snap(9000.0), COORD_MAX);
    }

    #[test]
    fn deltas_are_a_separate_block_from_coordinates() {
        let v = vocab();
        let d = v.delta(10.0);
        assert_eq!(v.delta_value(d), Some(10));
        // A delta id must not read as a coordinate, or boldening
        // would move points to absolute positions near the origin.
        assert_eq!(v.coord_value(d), None);
        let c = v.coord(10.0);
        assert_eq!(v.delta_value(c), None);
    }

    #[test]
    fn ops_round_trip() {
        let v = vocab();
        let ops = vec![
            Op::MoveTo(0.0, 0.0),
            Op::LineTo(100.0, 0.0),
            Op::CurveTo([(120.0, 0.0), (140.0, 20.0), (140.0, 40.0)]),
            Op::ClosePath,
        ];
        let ids = v.encode_ops(&ops);
        assert_eq!(v.decode_ops(&ids), ops);
    }

    #[test]
    fn a_truncated_run_decodes_as_far_as_it_got() {
        // Models stop mid-command. The caller should still get the
        // contour that was complete.
        let v = vocab();
        let mut ids = v.encode_ops(&[Op::MoveTo(0.0, 0.0), Op::LineTo(10.0, 10.0)]);
        ids.push(v.special("CURVE").unwrap());
        ids.push(v.coord(4.0));
        let ops = v.decode_ops(&ids);
        assert_eq!(ops, vec![Op::MoveTo(0.0, 0.0), Op::LineTo(10.0, 10.0)]);
    }

    #[test]
    fn an_unknown_glyph_is_an_error_not_a_guess() {
        assert!(matches!(vocab().name("thorn"), Err(Error::UnknownGlyph(_))));
    }

    #[test]
    fn an_unmapped_unicode_falls_back_to_none() {
        let v = vocab();
        let none = v.unicode(None).unwrap();
        assert_eq!(v.unicode(Some(0x9999)), Some(none));
        assert_ne!(v.unicode(Some(0x41)), Some(none));
    }
}
