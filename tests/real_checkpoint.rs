//! Runs against a real trained checkpoint when one is present.
//!
//! Set FONT_ML_TEST_MODEL to a model directory. Without it these
//! skip, so the suite still passes on a machine that has no
//! checkpoints (CI, a fresh clone).

use font_ml::{outline::OutlineModel, Checkpoint, ModelKind};

fn model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("FONT_ML_TEST_MODEL").map(Into::into)
}

#[test]
fn a_real_checkpoint_loads_and_runs() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: set FONT_ML_TEST_MODEL to a model directory");
        return;
    };

    let ckpt = Checkpoint::open(&dir).expect("open the model directory");
    assert_eq!(ckpt.config.kind, ModelKind::Outline);
    assert!(ckpt.config.dims > 0 && ckpt.config.layers > 0);
    assert!(!ckpt.glyph_names.is_empty(), "vocab.txt yielded no glyphs");
    eprintln!("{}", ckpt.summary());

    let model = OutlineModel::load(&ckpt).expect("load the weights");

    // The vocabulary this crate builds must be the size the checkpoint
    // was trained with, or every id means something else.
    assert_eq!(
        model.vocab().len(),
        ckpt.config.vocab_size,
        "vocabulary size disagrees with the checkpoint"
    );

    // A forward pass over a header-shaped prefix.
    let v = model.vocab();
    let name = ckpt.glyph_names.iter().find(|n| *n == "A")
        .cloned()
        .unwrap_or_else(|| ckpt.glyph_names[0].clone());
    let ids: Vec<u32> = vec![
        v.special("BOS").unwrap() as u32,
        v.name(&name).unwrap() as u32,
        v.weight(400) as u32,
    ];
    let logits = model.forward(&ids).expect("forward pass");
    assert_eq!(logits.dims(), &[ids.len(), ckpt.config.vocab_size]);

    // Finite logits: NaN here means the weights loaded into the wrong
    // shapes and every downstream number is meaningless.
    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|x| x.is_finite()), "logits contain NaN or inf");

    let next = model.next_token(&ids).expect("greedy step");
    eprintln!(
        "after BOS {name} W400 the model predicts {:?}",
        v.token(next as usize)
    );
}

/// Bolden a real glyph from a real font with a real checkpoint.
///
/// Needs both FONT_ML_TEST_MODEL and FONT_ML_TEST_UFO.
#[test]
fn boldening_a_real_glyph_keeps_it_compatible() {
    let (Some(dir), Some(ufo)) = (model_dir(), ufo_dir()) else {
        eprintln!("skipped: set FONT_ML_TEST_MODEL and FONT_ML_TEST_UFO");
        return;
    };
    let ckpt = Checkpoint::open(&dir).expect("open model");
    let model = OutlineModel::load(&ckpt).expect("load weights");

    let font = norad::Font::load(&ufo).expect("load ufo");
    let name = norad::Name::new("H").unwrap();
    let glyph = font.default_layer().get_glyph(&name).expect("glyph H");

    let ops = font_ml::ufo::glyph_ops(glyph).expect("H is a drawn glyph");
    assert!(!ops.is_empty());

    let center = ckpt
        .config
        .delta_center
        .map(|c| (c[0], c[1]))
        .unwrap_or((0, 0));
    let result = font_ml::bolden::bolden(
        &model,
        "H",
        Some(0x48),
        glyph.width,
        &ops,
        center,
    )
    .expect("bolden");

    assert!(
        result.is_compatible(),
        "the predicted master is not point-compatible with the source"
    );
    assert_eq!(result.deltas.len(), count_points(&ops));

    let moved = result.deltas.iter().filter(|(x, y)| *x != 0 || *y != 0).count();
    eprintln!(
        "H: {} points, {} moved, advance {:+}",
        result.deltas.len(),
        moved,
        result.advance_delta
    );
    let dx: Vec<i32> = result.deltas.iter().map(|d| d.0).collect();
    eprintln!("x offsets: {dx:?}");
    assert!(moved > 0, "every point stayed put; the model did nothing");
}

fn ufo_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("FONT_ML_TEST_UFO").map(Into::into)
}

fn count_points(ops: &[font_ml::tokenizer::Op]) -> usize {
    ops.iter()
        .map(|op| match op {
            font_ml::tokenizer::Op::CurveTo(_) => 3,
            font_ml::tokenizer::Op::ClosePath => 0,
            _ => 1,
        })
        .sum()
}
