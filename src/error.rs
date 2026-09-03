//! One error type, so callers can match rather than parse strings.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model directory {0} has no {1}")]
    MissingFile(PathBuf, &'static str),

    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} is not valid JSON: {source}")]
    Config {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("model is a {found} model; this operation needs a {wanted} model")]
    WrongKind { found: String, wanted: String },

    #[error(
        "the checkpoint and the vocabulary disagree: \
             config says vocab_size {config}, vocab.txt yields {vocab}"
    )]
    VocabMismatch { config: usize, vocab: usize },

    #[error("glyph {0} is not in this model's vocabulary")]
    UnknownGlyph(String),

    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
}
