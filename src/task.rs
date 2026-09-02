//! What a model can be asked to do.
//!
//! A font is not only a pile of glyph outlines, and the interesting
//! models are not all glyph models. Kerning is a property of a *pair*;
//! spacing is a property of a glyph in the company of others; feature
//! code is a property of the whole font. Naming the tasks separately
//! keeps the crate honest about which of them a given checkpoint can
//! actually do, so a caller, and an agent in particular, can ask
//! rather than assume.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A job a model may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
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

/// What kind of value an input or output is. A caller building a
/// form, a node, or a command line maps each kind to one widget or
/// one flag, and never has to know the task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// A UFO directory on disk.
    Source,
    /// A model directory on disk.
    Model,
    /// One glyph name from the source.
    Glyph,
    /// Zero or more glyph names, or every drawn glyph.
    Glyphs,
    /// A number with a default.
    Number,
    /// A yes or no.
    Flag,
    /// Free text.
    Text,
    /// A UFO layer written into the source.
    Layer,
    /// Per-glyph rows in the JSON report.
    Rows,
}

/// One thing a task takes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Input {
    /// The name, which is also the command-line flag (`--name`).
    pub name: String,
    /// What it is.
    pub kind: Kind,
    /// Whether a call without it is a usage error.
    pub required: bool,
    /// The default, for a number or a flag.
    pub default: Option<f64>,
    /// What it does, one line.
    pub help: String,
}

/// One thing a task produces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Output {
    /// The name. For a layer, the layer's name in the UFO.
    pub name: String,
    /// What it is.
    pub kind: Kind,
    /// What it holds, one line.
    pub help: String,
}

/// A task as a caller sees it: what it is, whether it runs today,
/// what it takes, and what it leaves behind. `font-ml tasks --json`
/// prints one of these per task, with the JSON Schema for the type,
/// so an editor or a node graph builds its controls from the list and
/// never carries a task name of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Spec {
    /// The task name, as `run` takes it.
    pub name: String,
    /// One line for a label.
    pub title: String,
    /// A few lines for a tooltip.
    pub help: String,
    /// Whether this build runs it.
    pub implemented: bool,
    /// What it takes, in the order a form would show them.
    pub inputs: Vec<Input>,
    /// What it produces.
    pub outputs: Vec<Output>,
}

fn input(name: &str, kind: Kind, required: bool, default: Option<f64>, help: &str) -> Input {
    Input {
        name: name.into(),
        kind,
        required,
        default,
        help: help.into(),
    }
}

fn output(name: &str, kind: Kind, help: &str) -> Output {
    Output {
        name: name.into(),
        kind,
        help: help.into(),
    }
}

impl Task {
    /// One line for a button or a node title.
    pub fn title(self) -> &'static str {
        match self {
            Task::Bolden => "Bolden",
            Task::Complete => "Complete a glyph",
            Task::Generate => "Generate a glyph",
            Task::Spacing => "Propose spacing",
            Task::Kerning => "Propose kerning",
            Task::Field => "Distance field",
        }
    }

    /// A few lines for a tooltip.
    pub fn help(self) -> &'static str {
        match self {
            Task::Bolden => {
                "Predict a heavier master from this one. Every point moves and none \
                 is added, so the result stays point-compatible."
            }
            Task::Complete => "Continue a partly drawn glyph.",
            Task::Generate => "Draw a glyph from its name and conditioning alone.",
            Task::Spacing => "Propose sidebearings for a glyph.",
            Task::Kerning => "Propose kerning values for glyph pairs.",
            Task::Field => "Render a glyph as a signed distance field.",
        }
    }

    /// The full description a caller builds its controls from.
    pub fn spec(self) -> Spec {
        let (inputs, outputs) = match self {
            Task::Bolden => (
                vec![
                    input("source", Kind::Source, true, None, "The UFO to read from."),
                    input("model", Kind::Model, true, None, "The model directory."),
                    input(
                        "glyph",
                        Kind::Glyphs,
                        false,
                        None,
                        "Which glyphs. --all for every drawn glyph.",
                    ),
                    input(
                        "strength",
                        Kind::Number,
                        false,
                        Some(1.0),
                        "Scale the predicted offsets.",
                    ),
                    input(
                        "reference",
                        Kind::Source,
                        false,
                        None,
                        "The other master; predictions are refitted to the weight it carries.",
                    ),
                    input(
                        "write",
                        Kind::Flag,
                        false,
                        Some(0.0),
                        "Write the predictions into the source as a proposal layer.",
                    ),
                ],
                vec![
                    output(
                        "com.runebender.proposal.bolden",
                        Kind::Layer,
                        "The predicted glyphs, next to the foreground, when --write is given.",
                    ),
                    output(
                        "glyphs",
                        Kind::Rows,
                        "Per glyph: points, points moved, advance delta, fitted strength.",
                    ),
                ],
            ),
            Task::Complete => (
                vec![
                    input("source", Kind::Source, true, None, "The UFO to read from."),
                    input("glyph", Kind::Glyph, true, None, "The glyph to continue."),
                    input(
                        "prefix",
                        Kind::Text,
                        true,
                        None,
                        "How much of the glyph is drawn.",
                    ),
                ],
                vec![output(
                    "com.runebender.proposal.complete",
                    Kind::Layer,
                    "The completed glyph.",
                )],
            ),
            Task::Generate => (
                vec![input(
                    "glyph",
                    Kind::Glyph,
                    true,
                    None,
                    "The glyph to draw.",
                )],
                vec![output(
                    "com.runebender.proposal.generate",
                    Kind::Layer,
                    "The drawn glyph.",
                )],
            ),
            Task::Spacing => (
                vec![
                    input("source", Kind::Source, true, None, "The UFO to read from."),
                    input("glyph", Kind::Glyph, true, None, "The glyph to space."),
                ],
                vec![output(
                    "glyphs",
                    Kind::Rows,
                    "Proposed sidebearings per glyph.",
                )],
            ),
            Task::Kerning => (
                vec![
                    input("source", Kind::Source, true, None, "The UFO to read from."),
                    input(
                        "left",
                        Kind::Glyph,
                        true,
                        None,
                        "The left glyph of the pair.",
                    ),
                    input(
                        "right",
                        Kind::Glyph,
                        true,
                        None,
                        "The right glyph of the pair.",
                    ),
                ],
                vec![output("pairs", Kind::Rows, "Proposed kerning per pair.")],
            ),
            Task::Field => (
                vec![input(
                    "glyph",
                    Kind::Glyph,
                    true,
                    None,
                    "The glyph to render.",
                )],
                vec![output("field", Kind::Rows, "The distance grid.")],
            ),
        };
        Spec {
            name: self.as_str().into(),
            title: self.title().into(),
            help: self.help().into(),
            implemented: self.implemented(),
            inputs,
            outputs,
        }
    }

    /// Every task's spec, in `all` order.
    pub fn specs() -> Vec<Spec> {
        Task::all().iter().map(|t| t.spec()).collect()
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
                let names: Vec<_> = Task::all().iter().map(|t| t.as_str()).collect();
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
    fn a_spec_names_its_inputs_the_way_run_takes_them() {
        // `inputs()` is the short list; the spec is the long one. They
        // must agree, or a caller reading one builds a call the other
        // rejects.
        for task in Task::all() {
            let spec = task.spec();
            for short in task.inputs() {
                assert!(
                    spec.inputs.iter().any(|i| i.name == *short),
                    "{task}: spec lacks input {short}"
                );
            }
            assert_eq!(spec.implemented, task.implemented());
        }
    }

    #[test]
    fn the_spec_has_a_schema_and_round_trips() {
        let schema = schemars::schema_for!(Spec);
        let text = serde_json::to_string(&schema).unwrap();
        assert!(text.contains("\"inputs\""));
        let spec = Task::Bolden.spec();
        let json = serde_json::to_string(&spec).unwrap();
        let back: Spec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
        assert!(json.contains("\"kind\":\"layer\""));
    }

    #[test]
    fn kerning_takes_a_pair_not_a_glyph() {
        // Kerning is a property of two glyphs together. If this ever
        // reads like the others, the model is answering a different
        // question from the one being asked.
        assert_eq!(Task::Kerning.inputs(), &["source", "left", "right"]);
    }
}
