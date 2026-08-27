//! Run small local models over font sources.
//!
//! This crate is not tied to any editor. It reads and writes UFO
//! sources, takes a model as a directory of files on disk, and runs
//! it. A person drives it from the command line, an editor embeds it,
//! an agent calls it as a library or shells out to the CLI and reads
//! JSON back. All three get the same behaviour.
//!
//! # Two kinds of model
//!
//! Font models come in two shapes, and both belong here because they
//! answer different questions.
//!
//! **Outline models** ([`ModelKind::Outline`]) predict drawing
//! commands: move, line, curve, close, with coordinates on a grid. The
//! output *is* an outline, so it can be dropped into a source and
//! edited. Because a prediction can be constrained to move existing
//! points rather than invent new ones, an outline model can hold a
//! glyph point-compatible with another master, which is what
//! interpolation requires.
//!
//! **Field models** ([`ModelKind::Field`]) predict a signed distance
//! field: a grid of distances to the nearest edge. Errors degrade
//! gracefully, which suits shapes with no fixed point structure to
//! preserve, such as the stacked, nonlinear composition of Nasta'liq.
//! The output has to be traced before it is editable, and tracing does
//! not preserve point structure.
//!
//! Neither replaces the other. Pick by whether the task has an outline
//! structure that must survive.
//!
//! # Models are directories
//!
//! A model is a directory holding `config.json`, `weights.safetensors`
//! and, for outline models, `vocab.txt`. Nothing is downloaded and
//! nothing phones home: you point at a folder and it loads.

pub mod bolden;
pub mod checkpoint;
pub mod error;
pub mod outline;
pub mod task;
pub mod tokenizer;
pub mod ufo;

pub use checkpoint::{Checkpoint, ModelConfig, ModelKind};
pub use error::{Error, Result};
pub use task::Task;
