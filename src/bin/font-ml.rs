//! The font-ml command line.
//!
//! Built to be driven by a person or by a program. That means: every
//! command takes `--json` and prints one machine-readable object;
//! nothing prompts; nothing needs a terminal; exit codes are stable
//! and distinguish "you asked wrongly" from "it is not built yet";
//! and `describe` reports capability, so a caller can find out what a
//! model can do instead of guessing and parsing an error.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use font_ml::{Checkpoint, Task};

/// Exit codes, stable across versions so a script can branch on them.
mod exit {
    pub const OK: u8 = 0;
    /// The request did not make sense: bad path, unknown task.
    pub const USAGE: u8 = 2;
    /// The request was understood and is not implemented yet.
    pub const UNIMPLEMENTED: u8 = 3;
    /// The request was understood and failed.
    pub const FAILED: u8 = 4;
}

#[derive(Parser)]
#[command(
    name = "font-ml",
    about = "Run small local models over font sources",
    long_about = "Run small local models over font sources.\n\n\
                  Models are directories holding config.json, \
                  weights.safetensors and, for outline models, \
                  vocab.txt. Nothing is downloaded.\n\n\
                  Every command accepts --json for machine-readable \
                  output."
)]
struct Cli {
    /// Print one JSON object instead of prose.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List the model directories under a path.
    Models {
        /// Directory to search. Its immediate children are checked.
        dir: PathBuf,
    },
    /// Report what a model is and which tasks it supports.
    Describe {
        /// The model directory.
        model: PathBuf,
    },
    /// List every task this tool knows about, and whether it works.
    Tasks,
    /// Score a model against a master somebody drew.
    Eval {
        /// The model directory.
        #[arg(long)]
        model: PathBuf,
        /// The lighter master.
        #[arg(long)]
        regular: PathBuf,
        /// The heavier master, drawn by hand.
        #[arg(long)]
        bold: PathBuf,
        /// Glyphs to score. Defaults to every drawn glyph in both.
        #[arg(long, value_delimiter = ',')]
        glyphs: Option<Vec<String>>,
        /// Stop after this many glyphs.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Scale predicted offsets before scoring.
        #[arg(long, default_value = "1.0")]
        strength: f64,
        /// Fit the strength per glyph so stems land on the weight the
        /// heavier master already carries, instead of using --strength.
        #[arg(long)]
        fit_stems: bool,
    },
    /// Run a task.
    Run {
        /// Task name, as listed by `tasks`.
        task: String,
        /// The model directory.
        #[arg(long)]
        model: PathBuf,
        /// The UFO to read the glyph from.
        #[arg(long)]
        source: Option<PathBuf>,
        /// Which glyph to run on.
        #[arg(long)]
        glyph: Option<String>,
        /// Scale the predicted offsets. Above 1 boldens harder.
        #[arg(long, default_value = "1.0")]
        strength: f64,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let code = match cli.command {
        Command::Models { dir } => models(&dir, cli.json),
        Command::Describe { model } => describe(&model, cli.json),
        Command::Tasks => {
            tasks(cli.json);
            exit::OK
        }
        Command::Eval { model, regular, bold, glyphs, limit, strength, fit_stems } => {
            eval(&model, &regular, &bold, glyphs, limit, strength, fit_stems, cli.json)
        }
        Command::Run { task, model, source, glyph, strength } => run(
            &task,
            &model,
            source.as_deref(),
            glyph.as_deref(),
            strength,
            cli.json,
        ),
    };
    ExitCode::from(code)
}

fn models(dir: &PathBuf, json: bool) -> u8 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return fail(json, exit::USAGE, &format!("cannot read {}", dir.display()));
    };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // Opening is cheap: it reads the config, not the weights.
        if let Ok(ckpt) = Checkpoint::open(&path) {
            found.push(ckpt);
        }
    }
    found.sort_by_key(|c| c.dir.clone());
    if json {
        let items: Vec<_> = found
            .iter()
            .map(|c| {
                serde_json::json!({
                    "path": c.dir,
                    "kind": c.config.kind.as_str(),
                    "dims": c.config.dims,
                    "layers": c.config.layers,
                    "glyphs": c.glyph_names.len(),
                })
            })
            .collect();
        println!("{}", serde_json::json!({ "models": items }));
    } else if found.is_empty() {
        println!("no models under {}", dir.display());
    } else {
        for c in &found {
            println!("{}", c.summary());
        }
    }
    exit::OK
}

fn describe(model: &PathBuf, json: bool) -> u8 {
    let ckpt = match Checkpoint::open(model) {
        Ok(c) => c,
        Err(e) => return fail(json, exit::USAGE, &e.to_string()),
    };
    // Which tasks this kind of model could answer at all.
    let supported: Vec<Task> = Task::all()
        .iter()
        .copied()
        .filter(|t| match ckpt.config.kind {
            font_ml::ModelKind::Outline => !matches!(t, Task::Field),
            font_ml::ModelKind::Field => matches!(t, Task::Field),
        })
        .collect();

    if json {
        println!(
            "{}",
            serde_json::json!({
                "path": ckpt.dir,
                "kind": ckpt.config.kind.as_str(),
                "dims": ckpt.config.dims,
                "layers": ckpt.config.layers,
                "heads": ckpt.config.heads,
                "vocab_size": ckpt.config.vocab_size,
                "max_len": ckpt.config.max_len,
                "glyphs": ckpt.glyph_names.len(),
                "tasks": supported.iter().map(|t| serde_json::json!({
                    "name": t.as_str(),
                    "implemented": t.implemented(),
                    "inputs": t.inputs(),
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        println!("{}", ckpt.summary());
        println!("  vocabulary  {} tokens", ckpt.config.vocab_size);
        println!("  context     {} tokens", ckpt.config.max_len);
        println!("  tasks:");
        for t in supported {
            let state = if t.implemented() { "ready" } else { "not built yet" };
            println!("    {:<10} {state}", t.as_str());
        }
    }
    exit::OK
}

fn tasks(json: bool) {
    if json {
        let items: Vec<_> = Task::all()
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.as_str(),
                    "implemented": t.implemented(),
                    "inputs": t.inputs(),
                })
            })
            .collect();
        println!("{}", serde_json::json!({ "tasks": items }));
    } else {
        for t in Task::all() {
            let state = if t.implemented() { "ready" } else { "not built yet" };
            println!("{:<10} {state:<14} needs: {}", t.as_str(), t.inputs().join(", "));
        }
    }
}

fn run(
    task: &str,
    model: &PathBuf,
    source: Option<&std::path::Path>,
    glyph: Option<&str>,
    strength: f64,
    json: bool,
) -> u8 {
    let task: Task = match task.parse() {
        Ok(t) => t,
        Err(e) => return fail(json, exit::USAGE, &e),
    };
    let checkpoint = match Checkpoint::open(model) {
        Ok(c) => c,
        Err(e) => return fail(json, exit::USAGE, &e.to_string()),
    };
    if !task.implemented() {
        return fail(
            json,
            exit::UNIMPLEMENTED,
            &format!(
                "{task} is not built yet; it will need: {}",
                task.inputs().join(", ")
            ),
        );
    }
    match task {
        Task::Bolden => bolden(&checkpoint, source, glyph, strength, json),
        _ => fail(json, exit::FAILED, "a ready task with no runner"),
    }
}

fn bolden(
    checkpoint: &Checkpoint,
    source: Option<&std::path::Path>,
    glyph: Option<&str>,
    strength: f64,
    json: bool,
) -> u8 {
    let (Some(source), Some(name)) = (source, glyph) else {
        return fail(json, exit::USAGE, "bolden needs --source and --glyph");
    };
    let font = match norad::Font::load(source) {
        Ok(f) => f,
        Err(e) => return fail(json, exit::USAGE, &format!("{source:?}: {e}")),
    };
    let Ok(key) = norad::Name::new(name) else {
        return fail(json, exit::USAGE, &format!("{name} is not a glyph name"));
    };
    let Some(g) = font.default_layer().get_glyph(&key) else {
        return fail(json, exit::USAGE, &format!("no glyph {name} in {source:?}"));
    };
    let Some(ops) = font_ml::ufo::glyph_ops(g) else {
        return fail(
            json,
            exit::USAGE,
            &format!("{name} has no outline to bolden; it may be a composite"),
        );
    };

    let model = match font_ml::outline::OutlineModel::load(checkpoint) {
        Ok(m) => m,
        Err(e) => return fail(json, exit::FAILED, &e.to_string()),
    };
    let center = checkpoint
        .config
        .delta_center
        .map(|c| (c[0], c[1]))
        .unwrap_or((0, 0));
    let unicode = g.codepoints.iter().next().map(|c| c as u32);
    let result = match font_ml::bolden::bolden(
        &model,
        name,
        unicode,
        g.width,
        &ops,
        center,
        checkpoint.config.trim_close,
        strength,
    ) {
        Ok(r) => r,
        Err(e) => return fail(json, exit::FAILED, &e.to_string()),
    };

    let moved = result
        .deltas
        .iter()
        .filter(|(x, y)| *x != 0 || *y != 0)
        .count();
    if json {
        println!(
            "{}",
            serde_json::json!({
                "glyph": name,
                "points": result.deltas.len(),
                "moved": moved,
                "advance_delta": result.advance_delta,
                "compatible": result.is_compatible(),
                "deltas": result.deltas,
            })
        );
    } else {
        println!(
            "{name}: {moved}/{} points moved, advance {:+}",
            result.deltas.len(),
            result.advance_delta
        );
    }
    exit::OK
}

fn eval(
    model_dir: &PathBuf,
    regular: &PathBuf,
    bold: &PathBuf,
    glyphs: Option<Vec<String>>,
    limit: usize,
    strength: f64,
    fit_stems: bool,
    json: bool,
) -> u8 {
    let checkpoint = match Checkpoint::open(model_dir) {
        Ok(c) => c,
        Err(e) => return fail(json, exit::USAGE, &e.to_string()),
    };
    let (Ok(reg_font), Ok(bold_font)) =
        (norad::Font::load(regular), norad::Font::load(bold))
    else {
        return fail(json, exit::USAGE, "could not load both masters");
    };
    let model = match font_ml::outline::OutlineModel::load(&checkpoint) {
        Ok(m) => m,
        Err(e) => return fail(json, exit::FAILED, &e.to_string()),
    };
    let center = checkpoint
        .config
        .delta_center
        .map(|c| (c[0], c[1]))
        .unwrap_or((0, 0));

    // Candidates: named, or every glyph drawn in both masters.
    let names: Vec<String> = match glyphs {
        Some(g) => g,
        None => reg_font
            .default_layer()
            .iter()
            .map(|g| g.name().to_string())
            .collect(),
    };

    let mut scores = Vec::new();
    let mut stems: Vec<(String, f64, f64)> = Vec::new();
    // The x-height from the heavier master, halved. Falling back to a
    // fraction of the em keeps this working on a source that never
    // filled in its metrics.
    let stem_height = bold_font
        .font_info
        .x_height
        .map(|v| v / 2.0)
        .unwrap_or_else(|| {
            bold_font
                .font_info
                .units_per_em
                .map(|v| *v * 0.25)
                .unwrap_or(256.0)
        });
    // The weight the heavier master already carries, from its own
    // drawn glyphs. Computed once at the same height everything else
    // is measured at: it is a property of the master, not of any one
    // prediction.
    let reference: Option<f64> = {
        let paths: Vec<_> = ["n", "i", "l", "h", "m", "u", "H", "I", "E"]
            .iter()
            .filter_map(|n| bold_font.default_layer().get_glyph(*n))
            .filter_map(font_ml::ufo::glyph_ops)
            .map(|ops| font_ml::stems::ops_to_path(&ops))
            .collect();
        font_ml::stems::reference_stem(&paths, stem_height)
    };
    for name in names {
        if scores.len() >= limit {
            break;
        }
        let Ok(key) = norad::Name::new(&name) else { continue };
        let (Some(rg), Some(bg)) = (
            reg_font.default_layer().get_glyph(&key),
            bold_font.default_layer().get_glyph(&key),
        ) else {
            continue;
        };
        let (Some(reg_ops), Some(bold_ops)) =
            (font_ml::ufo::glyph_ops(rg), font_ml::ufo::glyph_ops(bg))
        else {
            continue;
        };
        // Only comparable when the masters already agree structurally.
        if font_ml::eval::points(&reg_ops).len()
            != font_ml::eval::points(&bold_ops).len()
        {
            continue;
        }
        let unicode = rg.codepoints.iter().next().map(|c| c as u32);
        let Ok(result) = font_ml::bolden::bolden(
            &model,
            &name,
            unicode,
            rg.width,
            &reg_ops,
            center,
            checkpoint.config.trim_close,
            strength,
        ) else {
            continue;
        };
        // Re-run at a strength fitted to the weight already in the
        // heavier master, rather than a number somebody guessed.
        let result = if fit_stems {
            let fitted = reference.and_then(|target| {
                font_ml::stems::fit_strength(
                    &font_ml::stems::ops_to_path(&result.from),
                    &font_ml::stems::ops_to_path(&result.to),
                    target,
                    stem_height,
                )
            });
            match fitted.filter(|s| s.is_finite() && *s > 0.25 && *s < 4.0) {
                Some(s) => font_ml::bolden::bolden(
                    &model,
                    &name,
                    unicode,
                    rg.width,
                    &reg_ops,
                    center,
                    checkpoint.config.trim_close,
                    s,
                )
                .unwrap_or(result),
                None => result,
            }
        } else {
            result
        };
        scores.push(font_ml::eval::score(
            &name,
            &result.to,
            &bold_ops,
            &result.from,
            (center.0 as f64, center.1 as f64),
        ));
        // Half the x-height is where a lowercase stem is a stem and
        // not yet a join or a terminal.
        if let Some((p, a)) =
            font_ml::eval::stem_comparison(&result.to, &bold_ops, stem_height)
        {
            stems.push((name.clone(), p, a));
        }
    }

    if scores.is_empty() {
        return fail(json, exit::FAILED, "no comparable glyphs");
    }
    // Mean absolute stem error, and how many carry the right weight
    // within 4 units, which is about where a difference stops showing.
    let stem_mae = if stems.is_empty() {
        None
    } else {
        Some(stems.iter().map(|(_, p, a)| (p - a).abs()).sum::<f64>() / stems.len() as f64)
    };
    let stem_ok = stems.iter().filter(|(_, p, a)| (p - a).abs() <= 4.0).count();
    let mean = |f: fn(&font_ml::eval::Score) -> f64| -> f64 {
        scores.iter().map(f).sum::<f64>() / scores.len() as f64
    };
    let model_mae = mean(|s| s.model);
    let baseline_mae = mean(|s| s.baseline);
    let won = scores.iter().filter(|s| s.model < s.baseline).count();

    if json {
        println!(
            "{}",
            serde_json::json!({
                "glyphs": scores.len(),
                "model_mae": model_mae,
                "baseline_mae": baseline_mae,
                "beats_baseline": won,
                "stems_measured": stems.len(),
                "stem_mae": stem_mae,
                "stems_within_4_units": stem_ok,
                "per_glyph": scores.iter().map(|s| serde_json::json!({
                    "glyph": s.glyph,
                    "points": s.points,
                    "model": s.model,
                    "baseline": s.baseline,
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        println!(
            "{:>12} {:>8} {:>9} {:>8} {:>8}",
            "glyph", "model", "baseline", "stem", "wanted"
        );
        for s in &scores {
            let mark = if s.model < s.baseline { " " } else { " <-" };
            let stem = stems.iter().find(|(n, _, _)| *n == s.glyph);
            match stem {
                Some((_, p, a)) => println!(
                    "{:>12} {:>8.1} {:>9.1} {:>8.0} {:>8.0}{mark}",
                    s.glyph, s.model, s.baseline, p, a
                ),
                None => println!(
                    "{:>12} {:>8.1} {:>9.1} {:>8} {:>8}{mark}",
                    s.glyph, s.model, s.baseline, "-", "-"
                ),
            }
        }
        println!();
        println!(
            "{} glyphs: model {model_mae:.1}, baseline {baseline_mae:.1}, \
             model wins on {won}",
            scores.len()
        );
        match stem_mae {
            Some(mae) => println!(
                "{} stems measured: off by {mae:.1} units on average, \
                 {stem_ok} within 4",
                stems.len()
            ),
            None => println!("no stems could be measured at that height"),
        }
    }
    exit::OK
}

/// One shape of error output, so a caller parses one thing.
fn fail(json: bool, code: u8, message: &str) -> u8 {
    if json {
        println!("{}", serde_json::json!({ "error": message, "code": code }));
    } else {
        eprintln!("font-ml: {message}");
    }
    code
}
