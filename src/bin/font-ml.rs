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
    /// Run a task. Not implemented yet; reports what it would need.
    Run {
        /// Task name, as listed by `tasks`.
        task: String,
        /// The model directory.
        #[arg(long)]
        model: PathBuf,
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
        Command::Run { task, model } => run(&task, &model, cli.json),
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

fn run(task: &str, model: &PathBuf, json: bool) -> u8 {
    let task: Task = match task.parse() {
        Ok(t) => t,
        Err(e) => return fail(json, exit::USAGE, &e),
    };
    if let Err(e) = Checkpoint::open(model) {
        return fail(json, exit::USAGE, &e.to_string());
    }
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
    fail(json, exit::FAILED, "unreachable: a ready task with no runner")
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
