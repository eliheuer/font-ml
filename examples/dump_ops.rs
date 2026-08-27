//! Print the op sequence for one glyph, to compare against the
//! training lab's tokenizer.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ufo = args.next().expect("ufo path");
    let name = args.next().expect("glyph name");
    let font = norad::Font::load(&ufo)?;
    let key = norad::Name::new(&name)?;
    let g = font.default_layer().get_glyph(&key).expect("glyph");
    let ops = font_ml::ufo::glyph_ops(g).expect("drawn glyph");
    println!("width {} ops {}", g.width, ops.len());
    let show = |op: &font_ml::tokenizer::Op| match op {
        font_ml::tokenizer::Op::MoveTo(x, y) => format!("moveTo [({x}, {y})]"),
        font_ml::tokenizer::Op::LineTo(x, y) => format!("lineTo [({x}, {y})]"),
        font_ml::tokenizer::Op::CurveTo(p) => format!("curveTo {p:?}"),
        font_ml::tokenizer::Op::ClosePath => "closePath []".into(),
    };
    for op in ops.iter().take(6) {
        println!("  {}", show(op));
    }
    println!("  ...");
    let tail: Vec<_> = ops.iter().rev().take(3).collect();
    for op in tail.iter().rev() {
        println!("  {}", show(op));
    }
    Ok(())
}
