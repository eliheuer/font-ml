//! Loading a model from a directory.
//!
//! The layout is deliberately dull: a JSON config, a safetensors file,
//! and for outline models a newline-delimited vocabulary. It is what
//! the training lab already writes, and it is inspectable with `cat`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Which of the two model shapes this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Predicts drawing commands. Output is an outline.
    Outline,
    /// Predicts a signed distance field. Output is a grid of distances.
    Field,
}

impl ModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelKind::Outline => "outline",
            ModelKind::Field => "field",
        }
    }
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What the config file carries.
///
/// Checkpoints written before this crate existed have no `kind` field;
/// they are all outline models, so that is the default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_kind")]
    pub kind: ModelKind,
    pub dims: usize,
    pub layers: usize,
    pub heads: usize,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    /// Outline models trained on deltas record the centre the deltas
    /// are measured from.
    #[serde(default)]
    pub delta_center: Option<[i32; 2]>,
    #[serde(default)]
    pub trim_close: bool,
    /// Free-form, so a model can carry what the format does not name.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_kind() -> ModelKind {
    ModelKind::Outline
}

fn default_max_len() -> usize {
    1024
}

/// A model on disk, loaded but not yet run.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub dir: PathBuf,
    pub config: ModelConfig,
    /// Glyph names the model was trained on, in vocabulary order.
    /// Empty for field models.
    pub glyph_names: Vec<String>,
    /// Unicode values in the vocabulary, if it was built with them.
    pub unicodes: Vec<u32>,
}

impl Checkpoint {
    /// Read a model directory. Does not touch the weights: this is the
    /// cheap call a model picker makes over every folder it finds.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let config_path = dir.join("config.json");
        if !config_path.is_file() {
            return Err(Error::MissingFile(dir, "config.json"));
        }
        let text = std::fs::read_to_string(&config_path).map_err(|source| Error::Io {
            path: config_path.clone(),
            source,
        })?;
        let config: ModelConfig = serde_json::from_str(&text).map_err(|source| Error::Config {
            path: config_path,
            source,
        })?;

        if !dir.join("weights.safetensors").is_file() {
            return Err(Error::MissingFile(dir, "weights.safetensors"));
        }

        let (glyph_names, unicodes) = if config.kind == ModelKind::Outline {
            let vocab_path = dir.join("vocab.txt");
            if !vocab_path.is_file() {
                return Err(Error::MissingFile(dir, "vocab.txt"));
            }
            let raw = std::fs::read_to_string(&vocab_path).map_err(|source| Error::Io {
                path: vocab_path,
                source,
            })?;
            parse_vocab(&raw)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            dir,
            config,
            glyph_names,
            unicodes,
        })
    }

    pub fn weights_path(&self) -> PathBuf {
        self.dir.join("weights.safetensors")
    }

    /// A one-line description, for a model list in an interface.
    pub fn summary(&self) -> String {
        let name = self
            .dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| self.dir.display().to_string());
        format!(
            "{name}: {} model, {} layers x {} dims, {} glyphs",
            self.config.kind,
            self.config.layers,
            self.config.dims,
            self.glyph_names.len()
        )
    }

    /// Fail early when the caller wants the other kind of model.
    pub fn require(&self, wanted: ModelKind) -> Result<()> {
        if self.config.kind == wanted {
            Ok(())
        } else {
            Err(Error::WrongKind {
                found: self.config.kind.to_string(),
                wanted: wanted.to_string(),
            })
        }
    }
}

/// `vocab.txt` is one glyph name per line, with Unicode values on
/// `#U XXXX` lines after them.
fn parse_vocab(raw: &str) -> (Vec<String>, Vec<u32>) {
    let mut names = Vec::new();
    let mut unicodes = Vec::new();
    for line in raw.lines() {
        if line.is_empty() {
            continue;
        }
        if let Some(hex) = line.strip_prefix("#U ") {
            if let Ok(cp) = u32::from_str_radix(hex.trim(), 16) {
                unicodes.push(cp);
            }
        } else {
            names.push(line.to_string());
        }
    }
    (names, unicodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_splits_names_from_unicodes() {
        let (names, unis) = parse_vocab("A\nB\n#U 0041\n#U 0042\n");
        assert_eq!(names, ["A", "B"]);
        assert_eq!(unis, [0x41, 0x42]);
    }

    #[test]
    fn a_config_without_a_kind_is_an_outline_model() {
        // Every checkpoint written before this crate existed.
        let cfg: ModelConfig =
            serde_json::from_str(r#"{"dims":384,"layers":6,"heads":8,"vocab_size":1797}"#).unwrap();
        assert_eq!(cfg.kind, ModelKind::Outline);
        assert_eq!(cfg.max_len, 1024);
    }

    #[test]
    fn a_field_config_is_read_as_one() {
        let cfg: ModelConfig =
            serde_json::from_str(r#"{"kind":"field","dims":256,"layers":5,"heads":4}"#).unwrap();
        assert_eq!(cfg.kind, ModelKind::Field);
    }

    #[test]
    fn unknown_config_keys_are_kept_rather_than_rejected() {
        let cfg: ModelConfig =
            serde_json::from_str(r#"{"dims":8,"layers":1,"heads":1,"grid_w":16,"grid_h":14}"#)
                .unwrap();
        assert!(cfg.extra.contains_key("grid_w"));
    }
}
