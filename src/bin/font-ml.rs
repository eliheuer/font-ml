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
        /// An adapter directory, with `:strength` after it. Repeatable.
        #[arg(long)]
        adapter: Vec<String>,
    },
    /// Run a task.
    Run {
        /// Task name, as listed by `tasks`.
        task: String,
        /// The model directory. Every task but `train` needs one.
        #[arg(long)]
        model: Option<PathBuf>,
        /// The UFO to read glyphs from.
        #[arg(long)]
        source: Option<PathBuf>,
        /// Which glyph to run on. Repeat for several.
        #[arg(long)]
        glyph: Vec<String>,
        /// Every drawn glyph in the source.
        #[arg(long)]
        all: bool,
        /// Scale the predicted offsets. Above 1 boldens harder.
        #[arg(long, default_value = "1.0")]
        strength: f64,
        /// The other master, part-drawn. Where it says what weight it
        /// carries, each prediction is refitted to land there instead
        /// of on --strength.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Write the predictions into the source as a proposal layer
        /// (`com.runebender.proposal.<task>`), leaving the foreground
        /// alone. Without this, nothing is written.
        #[arg(long)]
        write: bool,
        /// No progress lines on stderr.
        #[arg(long)]
        quiet: bool,
        /// An adapter directory, with `:strength` after it. Repeat to
        /// stack; each adds to the weights in the order given.
        #[arg(long)]
        adapter: Vec<String>,
        /// train: write an adapter over --init into this directory
        /// instead of a whole model.
        #[arg(long)]
        adapter_out: Option<PathBuf>,
        /// train: the adapter's rank.
        #[arg(long, default_value = "8")]
        rank: usize,
        /// train: the heavier master.
        #[arg(long)]
        target: Option<PathBuf>,
        /// train: the model directory to write.
        #[arg(long)]
        out: Option<PathBuf>,
        /// train: optimizer steps.
        #[arg(long, default_value = "2000")]
        steps: usize,
        /// train: model width.
        #[arg(long, default_value = "384")]
        dims: usize,
        /// train: transformer blocks.
        #[arg(long, default_value = "6")]
        layers: usize,
        /// train: attention heads.
        #[arg(long, default_value = "8")]
        heads: usize,
        /// train: sequences per step.
        #[arg(long, default_value = "24")]
        batch: usize,
        /// train: stop after this many minutes; 0 runs every step.
        #[arg(long, default_value = "0")]
        minutes: f64,
        /// train: peak learning rate.
        #[arg(long, default_value = "0.0003")]
        lr: f64,
        /// train: mark colours that approve a target glyph: green,
        /// blue, or any.
        #[arg(long, default_value = "green", value_delimiter = ',')]
        colors: Vec<String>,
        /// train: centre the deltas on the corpus mean.
        #[arg(long)]
        recenter: bool,
        /// train: continue from this model directory.
        #[arg(long)]
        init: Option<PathBuf>,
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
        Command::Eval {
            model,
            regular,
            bold,
            glyphs,
            limit,
            strength,
            fit_stems,
            adapter,
        } => eval(
            &model, &regular, &bold, glyphs, limit, strength, fit_stems, &adapter, cli.json,
        ),
        Command::Run {
            task,
            source,
            quiet,
            target,
            out,
            steps,
            dims,
            layers,
            heads,
            batch,
            minutes,
            lr,
            colors,
            recenter,
            init,
            adapter_out,
            rank,
            ..
        } if task == "train" => train_cmd(
            source.as_deref(),
            target.as_deref(),
            out.as_deref(),
            font_ml::train::TrainConfig {
                steps,
                minutes,
                batch,
                dims,
                layers,
                heads,
                lr,
                recenter,
                colors,
                adapter_out,
                rank,
                ..Default::default()
            },
            init.as_deref(),
            quiet,
            cli.json,
        ),
        Command::Run {
            task,
            model,
            source,
            glyph,
            all,
            strength,
            reference,
            write,
            quiet,
            adapter,
            ..
        } => run(
            &task,
            model.as_ref(),
            source.as_deref(),
            RunOptions {
                glyphs: glyph,
                all,
                strength,
                reference,
                write,
                quiet,
                adapters: adapter,
            },
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
            let state = if t.implemented() {
                "ready"
            } else {
                "not built yet"
            };
            println!("    {:<10} {state}", t.as_str());
        }
    }
    exit::OK
}

fn tasks(json: bool) {
    if json {
        // The list and its schema together, so a caller can check
        // what it parsed against what was meant.
        println!(
            "{}",
            serde_json::json!({
                "tasks": Task::specs(),
                "schema": schemars::schema_for!(font_ml::task::Spec),
            })
        );
    } else {
        for t in Task::all() {
            let state = if t.implemented() {
                "ready"
            } else {
                "not built yet"
            };
            println!(
                "{:<10} {state:<14} needs: {}",
                t.as_str(),
                t.inputs().join(", ")
            );
        }
    }
}

/// What a `run` was asked to do beyond the task and the model.
struct RunOptions {
    glyphs: Vec<String>,
    all: bool,
    strength: f64,
    reference: Option<PathBuf>,
    write: bool,
    quiet: bool,
    adapters: Vec<String>,
}

/// `dir[:strength]` as an adapter to apply.
fn parse_adapters(specs: &[String]) -> Result<Vec<(font_ml::adapter::Adapter, f64)>, String> {
    specs
        .iter()
        .map(|spec| {
            let (dir, strength) = match spec.rsplit_once(':') {
                Some((d, s)) if !d.is_empty() && s.parse::<f64>().is_ok() => {
                    (d, s.parse::<f64>().unwrap_or(1.0))
                }
                _ => (spec.as_str(), 1.0),
            };
            font_ml::adapter::Adapter::open(dir)
                .map(|a| (a, strength))
                .map_err(|e| e.to_string())
        })
        .collect()
}

/// One line per glyph on stderr while a long run works, so a caller
/// that piped stderr can show a count and a person can see it move.
/// The shape is fixed: `progress <done>/<total> <glyph>`.
fn progress(quiet: bool, done: usize, total: usize, glyph: &str) {
    if !quiet && total > 1 {
        eprintln!("progress {done}/{total} {glyph}");
    }
}

fn run(
    task: &str,
    model: Option<&PathBuf>,
    source: Option<&std::path::Path>,
    options: RunOptions,
    json: bool,
) -> u8 {
    let task: Task = match task.parse() {
        Ok(t) => t,
        Err(e) => return fail(json, exit::USAGE, &e),
    };
    let Some(model) = model else {
        return fail(json, exit::USAGE, "--model is required");
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
        Task::Bolden => bolden(&checkpoint, source, &options, json),
        _ => fail(json, exit::FAILED, "a ready task with no runner"),
    }
}

/// `run train`: two masters in, a model directory out. Progress lines
/// carry the step and the loss; the JSON report carries the rest.
#[allow(clippy::too_many_arguments)]
fn train_cmd(
    source: Option<&std::path::Path>,
    target: Option<&std::path::Path>,
    out: Option<&std::path::Path>,
    cfg: font_ml::train::TrainConfig,
    init: Option<&std::path::Path>,
    quiet: bool,
    json: bool,
) -> u8 {
    let (Some(source), Some(target), Some(out)) = (source, target, out) else {
        return fail(
            json,
            exit::USAGE,
            "train needs --source (the lighter master), --target (the heavier) and --out",
        );
    };
    let mut on_progress = |step: usize, steps: usize, note: &str| {
        if !quiet {
            eprintln!("progress {step}/{steps} {note}");
        }
    };
    match font_ml::train::train(source, target, out, &cfg, init, &mut on_progress) {
        Ok(report) => {
            let is_adapter = cfg.adapter_out.is_some();
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "ok": true,
                        "model": if is_adapter { None } else { Some(&report.out) },
                        "adapter": if is_adapter { Some(&report.out) } else { None },
                        "steps": report.steps,
                        "best_val_loss": report.best_val,
                        "init_loss": report.init_loss,
                        "params": report.params,
                        "seconds": report.seconds,
                        "vocab": report.vocab,
                        "pairs": report.pairs,
                        "train_glyphs": report.train_glyphs,
                        "val_sequences": report.val_sequences,
                        "center": report.center,
                    })
                );
            } else {
                println!(
                    "{}: {} steps, best val loss {:.4}, {:.1}M params, {:.0}s",
                    report.out.display(),
                    report.steps,
                    report.best_val,
                    report.params as f64 / 1e6,
                    report.seconds
                );
            }
            exit::OK
        }
        Err(e) => fail(json, exit::FAILED, &e.to_string()),
    }
}

/// One glyph's prediction, as reported.
struct BoldenRow {
    glyph: String,
    points: usize,
    moved: usize,
    advance_delta: i32,
    fitted: Option<f64>,
    deltas: Vec<(i32, i32)>,
}

fn bolden(
    checkpoint: &Checkpoint,
    source: Option<&std::path::Path>,
    options: &RunOptions,
    json: bool,
) -> u8 {
    let Some(source) = source else {
        return fail(json, exit::USAGE, "bolden needs --source");
    };
    if options.glyphs.is_empty() && !options.all {
        return fail(json, exit::USAGE, "bolden needs --glyph, or --all");
    }
    let mut font = match norad::Font::load(source) {
        Ok(f) => f,
        Err(e) => return fail(json, exit::USAGE, &format!("{source:?}: {e}")),
    };
    let names: Vec<String> = if options.all {
        font.default_layer()
            .iter()
            .filter(|g| font_ml::ufo::glyph_ops(g).is_some())
            .map(|g| g.name().to_string())
            .collect()
    } else {
        options.glyphs.clone()
    };
    for name in &names {
        if norad::Name::new(name).is_err() {
            return fail(json, exit::USAGE, &format!("{name} is not a glyph name"));
        }
        if font.default_layer().get_glyph(name.as_str()).is_none() {
            return fail(json, exit::USAGE, &format!("no glyph {name} in {source:?}"));
        }
    }
    // The other master, where it says what weight it already carries.
    let reference = match &options.reference {
        Some(path) => match norad::Font::load(path) {
            Ok(other) => font_ml::stems::weight_delta(&font, &other),
            Err(e) => return fail(json, exit::USAGE, &format!("{path:?}: {e}")),
        },
        None => None,
    };

    let adapters = match parse_adapters(&options.adapters) {
        Ok(a) => a,
        Err(e) => return fail(json, exit::USAGE, &e),
    };
    let model = match font_ml::outline::OutlineModel::load_with_adapters(checkpoint, &adapters) {
        Ok(m) => m,
        Err(e) => return fail(json, exit::FAILED, &e.to_string()),
    };
    let center = checkpoint
        .config
        .delta_center
        .map(|c| (c[0], c[1]))
        .unwrap_or((0, 0));

    let mut rows: Vec<BoldenRow> = Vec::new();
    let mut proposed: Vec<norad::Glyph> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let total = names.len();
    for (done, name) in names.iter().enumerate() {
        progress(options.quiet, done + 1, total, name);
        let g = font
            .default_layer()
            .get_glyph(name.as_str())
            .expect("checked above");
        let Some(ops) = font_ml::ufo::glyph_ops(g) else {
            skipped.push((name.clone(), "no outline; a composite or empty".into()));
            continue;
        };
        let unicode = g.codepoints.iter().next().map(|c| c as u32);
        let predict = |strength: f64| {
            font_ml::bolden::bolden(
                &model,
                name,
                unicode,
                g.width,
                &ops,
                center,
                checkpoint.config.trim_close,
                strength,
            )
        };
        let mut result = match predict(options.strength) {
            Ok(r) => r,
            Err(e) => {
                skipped.push((name.clone(), e.to_string()));
                continue;
            }
        };
        // The model is better at shape than at weight, so where the
        // other master says what weight it carries, land there.
        let mut fitted = None;
        if let Some((delta, height)) = reference {
            let from_path = font_ml::stems::ops_to_path(&result.from);
            let want = font_ml::stems::target_from_delta(&from_path, delta, height)
                .and_then(|target| {
                    font_ml::stems::fit_strength(
                        &from_path,
                        &font_ml::stems::ops_to_path(&result.to),
                        target,
                        height,
                    )
                })
                .filter(|s| s.is_finite() && *s > 0.25 && *s < 4.0);
            if let Some(want) = want {
                if let Ok(refit) = predict(want) {
                    if refit.is_compatible() {
                        result = refit;
                        fitted = Some(want);
                    }
                }
            }
        }
        // The encoding guarantees this; check it before writing to a
        // font rather than take it on trust.
        let expected: usize = g.contours.iter().map(|c| c.points.len() + 1).sum();
        if !result.is_compatible() || result.deltas.len() != expected {
            skipped.push((
                name.clone(),
                format!(
                    "the prediction changed the point structure ({} offsets for {expected} points)",
                    result.deltas.len()
                ),
            ));
            continue;
        }
        let moved = result
            .deltas
            .iter()
            .filter(|(x, y)| *x != 0 || *y != 0)
            .count();
        if options.write {
            let contours = font_ml::ufo::apply_deltas(g, &result.deltas, center);
            proposed.push(font_ml::ufo::proposed_glyph(
                g,
                contours,
                result.advance_delta,
            ));
        }
        rows.push(BoldenRow {
            glyph: name.clone(),
            points: result.deltas.len(),
            moved,
            advance_delta: result.advance_delta,
            fitted,
            deltas: result.deltas,
        });
    }

    let layer = if options.write && !proposed.is_empty() {
        let layer = match font_ml::ufo::write_proposal(&mut font, "bolden", proposed) {
            Ok(l) => l,
            Err(e) => return fail(json, exit::FAILED, &format!("proposal layer: {e}")),
        };
        if let Err(e) = font.save(source) {
            return fail(json, exit::FAILED, &format!("{source:?}: {e}"));
        }
        Some(layer)
    } else {
        None
    };

    if rows.is_empty() {
        let why = skipped
            .first()
            .map(|(n, w)| format!("{n}: {w}"))
            .unwrap_or_else(|| "nothing to bolden".into());
        return fail(json, exit::FAILED, &why);
    }
    if json {
        // One glyph keeps the old flat shape, so a caller that read
        // `deltas` at the top level still can.
        let one = rows.len() == 1 && !options.all;
        let glyphs: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "glyph": r.glyph,
                    "points": r.points,
                    "moved": r.moved,
                    "advance_delta": r.advance_delta,
                    "compatible": true,
                    "fitted_strength": r.fitted,
                    "deltas": if one { serde_json::json!(r.deltas) } else { serde_json::Value::Null },
                })
            })
            .collect();
        let mut out = serde_json::json!({
            "ok": true,
            "task": "bolden",
            "glyphs": glyphs,
            "skipped": skipped,
            "proposal": layer.as_ref().map(|l| serde_json::json!({
                "layer": l, "glyphs": rows.len(), "source": source,
            })),
        });
        if one {
            let r = &rows[0];
            out["glyph"] = serde_json::json!(r.glyph);
            out["points"] = serde_json::json!(r.points);
            out["moved"] = serde_json::json!(r.moved);
            out["advance_delta"] = serde_json::json!(r.advance_delta);
            out["compatible"] = serde_json::json!(true);
            out["deltas"] = serde_json::json!(r.deltas);
        }
        println!("{out}");
    } else {
        for r in &rows {
            println!(
                "{}: {}/{} points moved, advance {:+}{}",
                r.glyph,
                r.moved,
                r.points,
                r.advance_delta,
                match r.fitted {
                    Some(s) => format!(", fitted to {s:.2}x"),
                    None => String::new(),
                }
            );
        }
        for (name, why) in &skipped {
            println!("{name}: skipped, {why}");
        }
        if let Some(layer) = &layer {
            println!("{} glyphs proposed in layer {layer}", rows.len());
        }
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
    adapter_specs: &[String],
    json: bool,
) -> u8 {
    let checkpoint = match Checkpoint::open(model_dir) {
        Ok(c) => c,
        Err(e) => return fail(json, exit::USAGE, &e.to_string()),
    };
    let (Ok(reg_font), Ok(bold_font)) = (norad::Font::load(regular), norad::Font::load(bold))
    else {
        return fail(json, exit::USAGE, "could not load both masters");
    };
    let adapters = match parse_adapters(adapter_specs) {
        Ok(a) => a,
        Err(e) => return fail(json, exit::USAGE, &e),
    };
    let model = match font_ml::outline::OutlineModel::load_with_adapters(&checkpoint, &adapters) {
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
    // How much weight the two masters differ by, learned from a few
    // glyphs drawn in both. Each glyph then adds that much to what it
    // already carries, rather than being pushed to one shared number,
    // which would flatten weights that are meant to differ.
    let reference: Option<f64> = {
        let pairs: Vec<_> = ["n", "o", "H", "O", "i", "l", "h", "m", "u", "I", "E"]
            .iter()
            .filter_map(|n| {
                let light = reg_font.default_layer().get_glyph(*n)?;
                let heavy = bold_font.default_layer().get_glyph(*n)?;
                Some((
                    font_ml::stems::ops_to_path(&font_ml::ufo::glyph_ops(light)?),
                    font_ml::stems::ops_to_path(&font_ml::ufo::glyph_ops(heavy)?),
                ))
            })
            .collect();
        font_ml::stems::reference_delta(&pairs, stem_height)
    };
    for name in names {
        if scores.len() >= limit {
            break;
        }
        let Ok(key) = norad::Name::new(&name) else {
            continue;
        };
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
        if font_ml::eval::points(&reg_ops).len() != font_ml::eval::points(&bold_ops).len() {
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
            let from_path = font_ml::stems::ops_to_path(&result.from);
            let fitted = reference.and_then(|delta| {
                let target = font_ml::stems::target_from_delta(&from_path, delta, stem_height)?;
                font_ml::stems::fit_strength(
                    &from_path,
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
        if let Some((p, a)) = font_ml::eval::stem_comparison(&result.to, &bold_ops, stem_height) {
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
    let stem_ok = stems
        .iter()
        .filter(|(_, p, a)| (p - a).abs() <= 4.0)
        .count();
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
