//! What a model can be asked to do.
//!
//! A font is not only a pile of glyph outlines, and the interesting
//! models are not all glyph models. Kerning is a property of a *pair*;
//! spacing is a property of a glyph in the company of others; feature
//! code is a property of the whole font. Naming the tasks separately
//! keeps the crate honest about which of them a given checkpoint can
//! actually do, so a caller, and an agent in particular, can ask
//! rather than assume.

use serde::{Deserialize, Serialize};

/// A job a model may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Task {
    /// Predict one master's outline from another's, as per-point
    /// offsets. The point structure is held fixed, so the result stays
    /// interpolation-compatible.
    Bolden,
    /// Continue a partly drawn glyph.
    Complete,
    /// Draw a glyph from its name and conditioning alone.
    Generate,
    /// Propose sidebearings for a glyph.
    Spacing,
    /// Propose kerning values for glyph pairs.
    Kerning,
    /// Render a glyph as a signed distance field.
    Field,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Bolden => "bolden",
            Task::Complete => "complete",
            Task::Generate => "generate",
            Task::Spacing => "spacing",
            Task::Kerning => "kerning",
            Task::Field => "field",
        }
    }

    /// Everything the crate knows how to name, whether or not any
    /// model implements it yet.
    pub fn all() -> &'static [Task] {
        &[
            Task::Bolden,
            Task::Complete,
            Task::Generate,
            Task::Spacing,
            Task::Kerning,
            Task::Field,
        ]
    }

    /// Whether this crate can run the task today. Reported by
    /// `describe`, so a caller finds out before it builds a pipeline
    /// on something that does not exist.
    pub fn implemented(self) -> bool {
        match self {
            Task::Bolden => true,
            // The pieces are in place: tokenizer, weights, forward
            // pass. Wiring them into an operation comes next.
            Task::Complete | Task::Generate => false,
            // No trained model for these yet, and no data pipeline.
            Task::Spacing | Task::Kerning => false,
            // Field models are declared in the format only.
            Task::Field => false,
        }
    }

    /// What the task needs as input, for a caller assembling a call.
    pub fn inputs(self) -> &'static [&'static str] {
        match self {
            Task::Bolden => &["source", "glyph"],
            Task::Complete => &["source", "glyph", "prefix"],
            Task::Generate => &["glyph"],
            Task::Spacing => &["source", "glyph"],
            Task::Kerning => &["source", "left", "right"],
            Task::Field => &["glyph"],
        }
    }
}

impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Task {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Task::all()
            .iter()
            .copied()
            .find(|t| t.as_str() == s)
            .ok_or_else(|| {
                let names: Vec<_> =
                    Task::all().iter().map(|t| t.as_str()).collect();
                format!("unknown task {s}; known tasks: {}", names.join(", "))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_round_trip_through_their_names() {
        for task in Task::all() {
            assert_eq!(task.as_str().parse::<Task>().unwrap(), *task);
        }
    }

    #[test]
    fn an_unknown_task_lists_the_known_ones() {
        let err = "hint".parse::<Task>().unwrap_err();
        assert!(err.contains("kerning"), "{err}");
    }

    #[test]
    fn what_is_reported_as_built_really_is() {
        // The point of reporting capability is that a caller can trust
        // it. A task marked implemented with no runner behind it is
        // worse than one honestly marked missing.
        assert!(Task::Bolden.implemented());
    }

    #[test]
    fn kerning_takes_a_pair_not_a_glyph() {
        // Kerning is a property of two glyphs together. If this ever
        // reads like the others, the model is answering a different
        // question from the one being asked.
        assert_eq!(Task::Kerning.inputs(), &["source", "left", "right"]);
    }
}
