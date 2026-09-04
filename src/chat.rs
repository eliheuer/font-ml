//! A chat model over the font, through the editor's own tools.
//!
//! The model is a quantized Qwen3 in a GGUF file, run by candle. What
//! it may do comes from `runebender-core agent tools`: a system prompt
//! and a tool list this crate never edits, so the pane in the editor
//! and a harness outside it drive the same commands. Each turn the
//! model writes, and when it writes a `<tool_call>` block the call
//! goes back to `runebender-core agent call` and the result returns
//! as the next message, up to a fixed number of calls. Nothing here
//! edits a font; the tools cannot either.
//!
//! Events go out as JSON lines on stdout, one per token or call, so a
//! caller streams a reply without parsing prose.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use crate::error::{Error, Result};

/// One message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    /// The text. For `tool`, the JSON result.
    pub content: String,
}

/// How a turn is generated.
#[derive(Debug, Clone)]
pub struct Options {
    /// Tokens per reply at most.
    pub max_tokens: usize,
    /// Tool calls per turn at most.
    pub max_calls: usize,
    /// Sampling temperature; 0 is greedy.
    pub temperature: f64,
    /// Nucleus threshold.
    pub top_p: f64,
    /// Seed for sampling.
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        // Qwen3's own recommendation for non-thinking use.
        Self {
            max_tokens: 1024,
            max_calls: 6,
            temperature: 0.7,
            top_p: 0.8,
            seed: 299_792_458,
        }
    }
}

/// A loaded model, ready to talk.
pub struct ChatModel {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    stops: Vec<u32>,
}

/// The device to run on. Kept for callers that only know "cpu or
/// not"; new code passes a [`crate::device::Choice`] to
/// [`crate::device::pick`].
pub fn device(cpu: bool) -> Result<Device> {
    crate::device::pick(if cpu {
        crate::device::Choice::Cpu
    } else {
        crate::device::Choice::Auto
    })
}

/// The GGUF and tokenizer for a model path: a directory holding
/// `model.gguf` and `tokenizer.json`, or the GGUF itself with the
/// tokenizer beside it.
fn files(path: &Path) -> Result<(PathBuf, PathBuf)> {
    if path.is_dir() {
        let gguf = std::fs::read_dir(path)
            .map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "gguf"))
            .ok_or_else(|| Error::MissingFile(path.to_path_buf(), "a .gguf file"))?;
        let tok = path.join("tokenizer.json");
        if !tok.is_file() {
            return Err(Error::MissingFile(path.to_path_buf(), "tokenizer.json"));
        }
        Ok((gguf, tok))
    } else {
        let dir = path.parent().unwrap_or(Path::new("."));
        let tok = dir.join("tokenizer.json");
        if !tok.is_file() {
            return Err(Error::MissingFile(dir.to_path_buf(), "tokenizer.json"));
        }
        Ok((path.to_path_buf(), tok))
    }
}

impl ChatModel {
    /// Reads the GGUF into memory on `device`. Seconds for a 4B model.
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let (gguf, tok) = files(path)?;
        let mut file = std::fs::File::open(&gguf).map_err(|source| Error::Io {
            path: gguf.clone(),
            source,
        })?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| Error::Tensor(e.with_path(&gguf)))?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;
        let tokenizer = Tokenizer::from_file(&tok).map_err(|e| Error::Io {
            path: tok.clone(),
            source: std::io::Error::other(e.to_string()),
        })?;
        let vocab = tokenizer.get_vocab(true);
        let stops: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| vocab.get(*t).copied())
            .collect();
        Ok(Self {
            model,
            tokenizer,
            device,
            stops,
        })
    }

    /// The conversation as Qwen3's chat template writes it, with
    /// thinking off so the reply starts at once. A tool result is a
    /// user turn holding a `<tool_response>` block, which is the
    /// form the model was trained on.
    pub fn render(messages: &[Message]) -> String {
        let mut out = String::new();
        for m in messages {
            match m.role.as_str() {
                "tool" => out.push_str(&format!(
                    "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n",
                    m.content
                )),
                role => out.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", m.content)),
            }
        }
        out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        out
    }

    /// One reply, streamed to `on_token` as text arrives. Stops at the
    /// end token, at `max_tokens`, or as soon as a tool block closes,
    /// so the call runs before the model talks past it. Returns the
    /// text and how many tokens it took.
    pub fn generate(
        &mut self,
        prompt: &str,
        options: &Options,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<(String, usize)> {
        self.model.clear_kv_cache();
        let encoded = self.tokenizer.encode(prompt, true).map_err(|e| Error::Io {
            path: PathBuf::from("prompt"),
            source: std::io::Error::other(e.to_string()),
        })?;
        let prompt_tokens = encoded.get_ids().to_vec();
        let sampling = if options.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopP {
                p: options.top_p,
                temperature: options.temperature,
            }
        };
        let mut sampler = LogitsProcessor::from_sampling(options.seed, sampling);

        // The prompt in chunks, then one token at a time. One pass over
        // a long prompt (a harness with thirty tools writes ten
        // thousand tokens) asks Metal for an attention buffer it will
        // not give; a few hundred tokens at a time always fits.
        const CHUNK: usize = 256;
        let mut logits = None;
        let mut offset = 0;
        for chunk in prompt_tokens.chunks(CHUNK) {
            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            logits = Some(self.model.forward(&input, offset)?.squeeze(0)?);
            offset += chunk.len();
        }
        let Some(logits) = logits else {
            return Ok((String::new(), 0));
        };
        let mut next = sampler.sample(&logits)?;
        let mut generated: Vec<u32> = vec![next];
        let mut text = String::new();
        let mut emitted = 0usize;
        let flush = |generated: &[u32],
                     text: &mut String,
                     emitted: &mut usize,
                     on_token: &mut dyn FnMut(&str)| {
            // Decode everything so far; emit what is new and whole.
            let decoded = self.tokenizer.decode(generated, true).unwrap_or_default();
            if decoded.len() > *emitted && !decoded.ends_with('\u{FFFD}') {
                let delta = &decoded[*emitted..];
                on_token(delta);
                text.push_str(delta);
                *emitted = decoded.len();
            }
        };
        flush(&generated, &mut text, &mut emitted, on_token);
        for index in 0..options.max_tokens {
            if self.stops.contains(&next) {
                break;
            }
            if text.contains("</tool_call>") {
                break;
            }
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, prompt_tokens.len() + index)?
                .squeeze(0)?;
            next = sampler.sample(&logits)?;
            generated.push(next);
            flush(&generated, &mut text, &mut emitted, on_token);
        }
        // Drop the end token's text if it slipped in.
        let clean = text
            .trim_end_matches("<|im_end|>")
            .trim_end_matches("<|endoftext|>")
            .to_string();
        Ok((clean, generated.len()))
    }
}

/// Where `runebender-core` is: `--core`, `$RUNEBENDER_CORE`, then PATH.
pub fn find_core(given: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = given {
        return Some(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("RUNEBENDER_CORE").filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(p));
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join("runebender-core"))
        .find(|c| c.is_file())
}

/// What `agent tools` said: the prompt and the tools.
pub fn harness(core: &Path) -> Result<(String, Vec<Value>)> {
    let output = std::process::Command::new(core)
        .arg("agent")
        .arg("tools")
        .output()
        .map_err(|source| Error::Io {
            path: core.to_path_buf(),
            source,
        })?;
    let value: Value = serde_json::from_slice(&output.stdout).map_err(|e| Error::Io {
        path: core.to_path_buf(),
        source: std::io::Error::other(format!("agent tools: {e}")),
    })?;
    let prompt = value
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tools = value
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok((prompt, tools))
}

/// Runs one call through core and returns its result object.
pub fn call_tool(core: &Path, font: &Path, name: &str, arguments: &Value) -> Value {
    let output = std::process::Command::new(core)
        .arg("agent")
        .arg("call")
        .arg(name)
        .arg("--font")
        .arg(font)
        .arg("--args")
        .arg(arguments.to_string())
        .output();
    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            stdout
                .lines()
                .rev()
                .find_map(|l| serde_json::from_str(l).ok())
                .unwrap_or_else(|| {
                    json!({ "name": name, "ok": false, "result": { "error": String::from_utf8_lossy(&o.stderr).trim() } })
                })
        }
        Err(e) => json!({ "name": name, "ok": false, "result": { "error": e.to_string() } }),
    }
}

/// The `<tool_call>` blocks in a reply. Mirrors core's parser, so a
/// caller without core still reads the same shape.
pub fn parse_tool_calls(text: &str) -> Vec<(String, Value)> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let (body, next) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        if let Ok(v) = serde_json::from_str::<Value>(body.trim()) {
            let name = v
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let args = v.get("arguments").cloned().unwrap_or(json!({}));
            if !name.is_empty() {
                calls.push((name, args));
            }
        }
        rest = next;
    }
    calls
}

/// A tool result trimmed to what a small context can hold.
fn shorten(value: &Value, limit: usize) -> String {
    let text = value.to_string();
    if text.len() <= limit {
        return text;
    }
    let cut: String = text.chars().take(limit).collect();
    format!("{cut}… (truncated, {} chars)", text.len())
}

/// What a turn reports at the end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnReport {
    /// The assistant's final text, tool blocks removed.
    pub text: String,
    /// The calls made, in order, with their results.
    pub calls: Vec<Value>,
    /// Tokens generated across the turn.
    pub tokens: usize,
    /// Wall time.
    pub seconds: f64,
    /// Generation speed over the turn.
    pub tokens_per_second: f64,
    /// The full message list after the turn, for the next one.
    pub messages: Vec<Message>,
}

/// Whether the person's question is about a glyph's geometry. The
/// same word list as core's `agent::asks_geometry`; kept here because
/// this crate does not link core, only runs it.
pub fn asks_geometry(question: &str) -> bool {
    let q = question.to_lowercase();
    [
        "wide",
        "width",
        "advance",
        "sidebearing",
        "side bearing",
        "lsb",
        "rsb",
        "spacing",
        "points",
        "contour",
        "anchor",
        "shape",
        "outline",
        "bounds",
        "how tall",
        "height of",
    ]
    .iter()
    .any(|w| q.contains(w))
}

/// Whether the question is about a format or a term, so documentation
/// is fetched before the model answers. Mirrors core's `asks_docs`.
pub fn asks_docs(question: &str) -> bool {
    let q = question.to_lowercase();
    [
        "spec",
        "ufo",
        "glif",
        "designspace",
        "fontc",
        "opentype",
        "attribute",
        "what does",
        "what is a",
        "what is the",
        "mean",
        "documentation",
    ]
    .iter()
    .any(|w| q.contains(w))
}

/// Sent once when the model answered a geometry question without
/// reading the glyph.
const READ_FIRST_NUDGE: &str = "You answered without reading the glyph. Call read_glyph on it \
                                now and answer only from the result.";

/// One assistant turn with its tool loop. `messages` holds the
/// conversation so far without a system message; the harness prompt
/// goes first. `emit` hears every event as JSON.
#[allow(clippy::too_many_arguments)]
pub fn turn(
    model: &mut ChatModel,
    core: &Path,
    font: &Path,
    prompt: &str,
    messages: &[Message],
    options: &Options,
    emit: &mut dyn FnMut(Value),
) -> Result<TurnReport> {
    let started = Instant::now();
    let mut convo: Vec<Message> = Vec::with_capacity(messages.len() + 4);
    convo.push(Message {
        role: "system".into(),
        content: prompt.to_string(),
    });
    convo.extend(messages.iter().cloned());
    let question = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let mut tokens = 0;
    let mut calls = Vec::new();
    let mut final_text = String::new();
    // A question about a spec or a term gets the documentation before
    // the model speaks, as a tool result it did not have to ask for.
    if asks_docs(&question) {
        let args = json!({ "query": question });
        emit(json!({ "event": "tool_call", "name": "docs", "arguments": args }));
        let result = call_tool(core, font, "docs", &args);
        emit(
            json!({ "event": "tool_result", "name": "docs", "ok": result.get("ok"), "result": result.get("result") }),
        );
        calls.push(result.clone());
        convo.push(Message {
            role: "tool".into(),
            content: shorten(&result, 6000),
        });
    }
    let mut nudged = false;
    for _ in 0..=options.max_calls {
        let rendered = ChatModel::render(&convo);
        let mut on_token = |t: &str| emit(json!({ "event": "token", "text": t }));
        let (reply, n) = model.generate(&rendered, options, &mut on_token)?;
        tokens += n;
        let tool_calls = parse_tool_calls(&reply);
        convo.push(Message {
            role: "assistant".into(),
            content: reply.clone(),
        });
        if tool_calls.is_empty() {
            // The guard: a geometry answer with no read behind it is
            // sent back once, with the rule restated.
            let read = calls.iter().any(|c| {
                matches!(
                    c.get("name").and_then(Value::as_str),
                    Some("read_glyph") | Some("proof")
                )
            });
            if asks_geometry(&question) && !read && !nudged {
                nudged = true;
                emit(json!({ "event": "nudge", "text": READ_FIRST_NUDGE }));
                convo.push(Message {
                    role: "user".into(),
                    content: READ_FIRST_NUDGE.to_string(),
                });
                continue;
            }
            final_text = strip(&reply);
            break;
        }
        if calls.len() >= options.max_calls {
            final_text = strip(&reply);
            break;
        }
        for (name, args) in tool_calls {
            emit(json!({ "event": "tool_call", "name": name, "arguments": args }));
            let result = call_tool(core, font, &name, &args);
            emit(
                json!({ "event": "tool_result", "name": name, "ok": result.get("ok"), "result": result.get("result") }),
            );
            calls.push(result.clone());
            convo.push(Message {
                role: "tool".into(),
                content: shorten(&result, 6000),
            });
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let report = TurnReport {
        text: final_text,
        calls,
        tokens,
        seconds,
        tokens_per_second: if seconds > 0.0 {
            tokens as f64 / seconds
        } else {
            0.0
        },
        messages: convo.into_iter().skip(1).collect(),
    };
    emit(
        json!({ "event": "done", "text": report.text, "tokens": tokens, "seconds": seconds, "tokens_per_second": report.tokens_per_second }),
    );
    Ok(report)
}

fn strip(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</tool_call>") {
            Some(end) => rest = &rest[start + end + "</tool_call>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Generates `n` tokens from a fixed prompt and reports the speed,
/// for comparing models and devices.
pub fn bench(model: &mut ChatModel, n: usize) -> Result<(usize, f64)> {
    let prompt = ChatModel::render(&[Message {
        role: "user".into(),
        content: "Describe the letter g in a sans serif typeface in detail.".into(),
    }]);
    let options = Options {
        max_tokens: n,
        temperature: 0.0,
        ..Default::default()
    };
    let started = Instant::now();
    let mut sink = |_: &str| {};
    let (_, tokens) = model.generate(&prompt, &options, &mut sink)?;
    let seconds = started.elapsed().as_secs_f64();
    let _ = std::io::stdout().flush();
    Ok((tokens, seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_template_ends_with_an_open_assistant_turn() {
        let r = ChatModel::render(&[Message {
            role: "user".into(),
            content: "hi".into(),
        }]);
        assert!(r.starts_with("<|im_start|>user\nhi<|im_end|>\n"));
        assert!(r.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn tool_results_are_user_turns_with_a_response_block() {
        let r = ChatModel::render(&[Message {
            role: "tool".into(),
            content: "{\"ok\":true}".into(),
        }]);
        assert!(r.contains("<tool_response>\n{\"ok\":true}\n</tool_response>"));
    }

    #[test]
    fn questions_are_sorted_by_what_they_need() {
        assert!(asks_geometry(
            "How wide is the H, and what are its sidebearings?"
        ));
        assert!(!asks_geometry(
            "What font is open and how many glyphs does it have?"
        ));
        assert!(asks_docs(
            "What does the UFO spec say about the smooth attribute on a point?"
        ));
        assert!(!asks_docs(
            "Propose a bolder H with the virtua-12m-bolden model"
        ));
    }

    #[test]
    fn calls_parse_and_strip() {
        let text = "Looking.\n<tool_call>\n{\"name\":\"font_info\",\"arguments\":{}}\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "font_info");
        assert_eq!(strip(text), "Looking.");
    }
}
