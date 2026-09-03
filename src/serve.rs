//! An OpenAI-compatible chat endpoint over the loaded model.
//!
//! `font-ml serve` keeps one model resident and answers
//! `POST /v1/chat/completions` on the loopback interface, so a harness
//! that speaks that protocol (OMP, or anything with a "custom OpenAI
//! provider" setting) drives a local model with no cloud and no other
//! runtime. Tools the caller declares are written into the prompt in
//! Qwen3's form and parsed back out of the reply as `tool_calls`;
//! tool results come back in as `role: tool` messages.
//!
//! The server is plain HTTP/1.1 over `std::net`, one request at a
//! time, because one laptop has one GPU and one model. No framework,
//! nothing to vet beyond what the crate already has.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

use serde_json::{json, Value};

use crate::chat::{parse_tool_calls, ChatModel, Message, Options};
use crate::error::{Error, Result};

/// Serves until the process ends. `bind` is `host:port`.
pub fn serve(model: &mut ChatModel, bind: &str, options: &Options) -> Result<()> {
    let listener = TcpListener::bind(bind).map_err(|source| Error::Io {
        path: std::path::PathBuf::from(bind),
        source,
    })?;
    eprintln!("font-ml serve: listening on http://{bind}/v1/chat/completions");
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        if let Err(e) = handle(model, stream, options) {
            eprintln!("font-ml serve: {e}");
        }
    }
    Ok(())
}

/// One request.
fn handle(model: &mut ChatModel, mut stream: TcpStream, options: &Options) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone().map_err(io_err)?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).map_err(io_err)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(io_err)?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).map_err(io_err)?;
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/models") => {
            let body =
                json!({ "object": "list", "data": [{ "id": "font-ml", "object": "model" }] });
            respond(
                &mut stream,
                200,
                "application/json",
                body.to_string().as_bytes(),
            )
        }
        ("POST", "/v1/chat/completions") => {
            let request: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => {
                    let err = json!({ "error": { "message": format!("bad JSON: {e}") } });
                    return respond(
                        &mut stream,
                        400,
                        "application/json",
                        err.to_string().as_bytes(),
                    );
                }
            };
            completion(model, &mut stream, &request, options)
        }
        _ => respond(&mut stream, 404, "text/plain", b"not found"),
    }
}

fn io_err(source: std::io::Error) -> Error {
    Error::Io {
        path: std::path::PathBuf::from("http"),
        source,
    }
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).map_err(io_err)?;
    stream.write_all(body).map_err(io_err)?;
    stream.flush().map_err(io_err)
}

/// The caller's tools, written into the system prompt the way the
/// harness in core writes its own, so Qwen3 emits the same block.
fn tools_prompt(tools: &[Value]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "\n\nTools. To call one, write exactly one block like this and nothing after \
         it, then wait for the result:\n<tool_call>\n{\"name\": \"tool_name\", \
         \"arguments\": {}}\n</tool_call>\n\n",
    );
    for t in tools {
        let f = t.get("function").unwrap_or(t);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        let desc = f.get("description").and_then(Value::as_str).unwrap_or("");
        let params = f
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .cloned()
            .unwrap_or(json!({}));
        out.push_str(&format!("- {name}: {desc} Arguments: {params}\n"));
    }
    out
}

/// The OpenAI messages as ours. A `tool` message becomes a tool
/// response; an assistant message with `tool_calls` is rebuilt as the
/// block the model wrote.
fn messages_from(request: &Value, tools: &[Value]) -> Vec<Message> {
    let mut out = Vec::new();
    let mut system = String::new();
    for m in request
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        match role {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&content);
            }
            "assistant" => {
                let mut text = content;
                if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                    for c in calls {
                        let f = c.get("function").unwrap_or(c);
                        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = f
                            .get("arguments")
                            .map(|a| match a {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .unwrap_or_else(|| "{}".into());
                        text.push_str(&format!(
                            "\n<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"
                        ));
                    }
                }
                out.push(Message {
                    role: "assistant".into(),
                    content: text,
                });
            }
            "tool" => out.push(Message {
                role: "tool".into(),
                content,
            }),
            _ => out.push(Message {
                role: "user".into(),
                content,
            }),
        }
    }
    let mut all = Vec::with_capacity(out.len() + 1);
    let prompt = format!("{system}{}", tools_prompt(tools));
    if !prompt.trim().is_empty() {
        all.push(Message {
            role: "system".into(),
            content: prompt,
        });
    }
    all.extend(out);
    all
}

/// One completion, streamed as SSE when asked, else one JSON body.
fn completion(
    model: &mut ChatModel,
    stream: &mut TcpStream,
    request: &Value,
    options: &Options,
) -> Result<()> {
    let tools: Vec<Value> = request
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let messages = messages_from(request, &tools);
    let prompt = ChatModel::render(&messages);
    let streaming = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let max_tokens = request
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map_or(options.max_tokens, |n| n as usize);
    let opts = Options {
        max_tokens,
        ..options.clone()
    };
    let id = format!("chatcmpl-{}", std::process::id());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if streaming {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        stream.write_all(head.as_bytes()).map_err(io_err)?;
        let mut sink = stream.try_clone().map_err(io_err)?;
        let mut in_tool = false;
        let mut on_token = |t: &str| {
            // Tool blocks are not content; they come at the end as
            // tool_calls. Everything before the first block streams.
            if in_tool || t.contains("<tool_call>") {
                in_tool = true;
                return;
            }
            let chunk = json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": "font-ml",
                "choices": [{ "index": 0, "delta": { "content": t }, "finish_reason": Value::Null }]
            });
            let _ = sink.write_all(format!("data: {chunk}\n\n").as_bytes());
            let _ = sink.flush();
        };
        let (text, _) = model.generate(&prompt, &opts, &mut on_token)?;
        let calls = parse_tool_calls(&text);
        let finish = if calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        let mut delta = json!({});
        if !calls.is_empty() {
            delta["tool_calls"] = json!(calls
                .iter()
                .enumerate()
                .map(|(i, (name, args))| json!({
                    "index": i, "id": format!("call_{i}"), "type": "function",
                    "function": { "name": name, "arguments": args.to_string() }
                }))
                .collect::<Vec<_>>());
        }
        let last = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": "font-ml",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
        });
        stream
            .write_all(format!("data: {last}\n\ndata: [DONE]\n\n").as_bytes())
            .map_err(io_err)?;
        stream.flush().map_err(io_err)
    } else {
        let mut sink = |_: &str| {};
        let (text, tokens) = model.generate(&prompt, &opts, &mut sink)?;
        let calls = parse_tool_calls(&text);
        let visible = strip(&text);
        let mut message = json!({ "role": "assistant", "content": if visible.is_empty() { Value::Null } else { Value::String(visible) } });
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls
                .iter()
                .enumerate()
                .map(|(i, (name, args))| json!({
                    "id": format!("call_{i}"), "type": "function",
                    "function": { "name": name, "arguments": args.to_string() }
                }))
                .collect::<Vec<_>>());
        }
        let body = json!({
            "id": id, "object": "chat.completion", "created": created, "model": "font-ml",
            "choices": [{ "index": 0, "message": message, "finish_reason": if calls.is_empty() { "stop" } else { "tool_calls" } }],
            "usage": { "prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens }
        });
        respond(stream, 200, "application/json", body.to_string().as_bytes())
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_messages_become_ours() {
        let req = json!({
            "messages": [
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "", "tool_calls": [{ "id": "c1", "type": "function", "function": { "name": "bash", "arguments": "{\"cmd\":\"ls\"}" } }] },
                { "role": "tool", "tool_call_id": "c1", "content": "a b c" }
            ],
            "tools": [{ "type": "function", "function": { "name": "bash", "description": "Run a command.", "parameters": { "type": "object", "properties": { "cmd": { "type": "string" } } } } }]
        });
        let tools = req["tools"].as_array().unwrap().clone();
        let m = messages_from(&req, &tools);
        assert_eq!(m[0].role, "system");
        assert!(m[0].content.contains("Be brief.") && m[0].content.contains("- bash:"));
        assert_eq!(m[2].role, "assistant");
        assert!(m[2].content.contains("<tool_call>"));
        assert_eq!(m[3].role, "tool");
    }
}
