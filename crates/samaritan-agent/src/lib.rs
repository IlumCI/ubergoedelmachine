//! Talking to a local model, and turning what it says into a decision record.
//!
//! One OpenAI-compatible HTTP client, pointed at whatever is serving on
//! localhost. Blocking rather than async on purpose: the concurrency that
//! matters is the inference server's parallel slots, driven by ordinary OS
//! threads here, and an async runtime would add a dependency without adding
//! a request per second.
//!
//! The important structural decision in this crate is [`DraftDecision`]. The
//! model emits a *draft* — situation, options, choice, prediction, actions —
//! and the harness supplies the rest: the identity, the policy version, and
//! the **authority**. A model that could set its own provenance could set it
//! to trusted, which is the same failure as letting it grade its own action
//! tiers. Provenance is computed from what was put into the prompt, by
//! [`prompt::Prompt::authority`], and the model never sees the field.

pub mod grammar;
pub mod prompt;

pub use grammar::{DRAFT_DECISION_GBNF, draft_decision_schema};

use std::time::Duration;

use samaritan_dsl::decision::PolicyVersion;
use samaritan_dsl::{
    ActionKind, Authority, BlastRadius, Confidence, DecisionId, DecisionOption, DecisionRecord,
    DslError, Prediction, ProposedAction, Reversibility,
};
use serde::{Deserialize, Serialize};

pub use prompt::Prompt;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("transport: {0}")]
    Transport(String),

    #[error("the server returned {status}: {body}")]
    Status { status: u16, body: String },

    #[error("the response was not a chat completion: {0}")]
    Shape(String),

    #[error("the model emitted something that is not a draft decision: {0}")]
    Parse(String),

    #[error("the draft did not satisfy the decision language: {0}")]
    Invalid(#[from] DslError),

    #[error("gave up after {attempts} attempts; last error: {last}")]
    Exhausted { attempts: u32, last: String },
}

/// How the server should be made to produce well-formed output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Constrain {
    /// Attach a GBNF grammar. llama.cpp only, and much the best option: the
    /// sampler cannot produce invalid output, so there is nothing to retry.
    Grammar,
    /// Attach a JSON schema via `response_format`. Weaker but portable.
    Schema,
    /// Ask nicely and parse what comes back. The fallback, and on an
    /// abliterated model an expensive one.
    None,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Base URL of an OpenAI-compatible server, e.g.
    /// `http://127.0.0.1:8080/v1`.
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub temperature: f64,
    /// Hard cap on generated tokens. At single-digit tokens per second this
    /// is a wall-clock budget as much as a length limit.
    pub max_tokens: u32,
    /// llama.cpp's n-gram repetition penalty. `1.0` is off; above it discourages
    /// repeating recent tokens.
    ///
    /// Not cosmetic here. The abliterated adversary, unpenalised, falls into
    /// repetition loops — the same clause or the same `..\` segment emitted
    /// hundreds of times — that overrun `max_tokens` mid-string and yield
    /// unparseable JSON, forcing a fallback. A mild penalty is what lets its
    /// most ambitious attacks finish as valid objects. Emitted on every request;
    /// servers that do not implement the key ignore it.
    pub repeat_penalty: f64,
    pub constrain: Constrain,
    pub timeout: Duration,
    /// Retries when the output will not parse. Zero is correct with
    /// [`Constrain::Grammar`], where a parse failure means this crate has a
    /// bug rather than that the model had a bad day.
    pub max_retries: u32,
    /// Base seed for sampling. `None` draws one per process.
    ///
    /// Borrowed from RustLMHub, whose server puts it well: a seed the caller
    /// cannot see is a seed that does not exist. Sampling is the only
    /// nondeterminism in an episode, so pinning it and writing it down is
    /// what makes a run reproducible — and a self-improvement result nobody
    /// can re-run is not a result.
    ///
    /// Chosen here rather than read back from the response, because not every
    /// OpenAI-compatible server echoes it, and a seed that is sometimes
    /// recorded is no better than one that never is.
    pub seed: Option<u64>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8080/v1".into(),
            api_key: String::new(),
            model: "local".into(),
            temperature: 0.7,
            max_tokens: 1024,
            repeat_penalty: 1.0,
            constrain: Constrain::Grammar,
            timeout: Duration::from_secs(600),
            max_retries: 2,
            seed: None,
        }
    }
}

/// What the model is allowed to say.
///
/// Note the absent fields: no `id`, no `policy_version`, no `authority`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DraftDecision {
    pub situation: String,
    pub options: Vec<DraftOption>,
    pub chosen: usize,
    pub rationale: String,
    pub prediction: DraftPrediction,
    #[serde(default)]
    pub actions: Vec<DraftAction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DraftOption {
    pub summary: String,
    pub assessment: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DraftPrediction {
    pub outcome: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DraftAction {
    pub kind: ActionKind,
    pub reversibility: Reversibility,
    pub blast_radius: BlastRadius,
    pub intent: String,
    pub payload: serde_json::Value,
}

impl DraftDecision {
    /// Complete the draft into a record, with the harness supplying
    /// everything the model is not trusted to state.
    pub fn seal(
        self,
        policy_version: PolicyVersion,
        authority: Authority,
    ) -> Result<DecisionRecord, DslError> {
        DecisionRecord::new(
            DecisionId::new(),
            self.situation,
            self.options
                .into_iter()
                .map(|o| DecisionOption {
                    summary: o.summary,
                    assessment: o.assessment,
                })
                .collect(),
            self.chosen,
            self.rationale,
            Prediction {
                outcome: self.prediction.outcome,
                confidence: Confidence::new(self.prediction.confidence)?,
            },
            self.actions
                .into_iter()
                .map(|a| ProposedAction {
                    kind: a.kind,
                    reversibility: a.reversibility,
                    blast_radius: a.blast_radius,
                    intent: a.intent,
                    payload: a.payload,
                })
                .collect(),
            policy_version,
            authority,
        )
    }
}

/// What a call cost, for the compute budget and the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Prompt tokens the server reported as already cached. The measurement
    /// that tells you whether the prompt layout is doing its job.
    pub cached_tokens: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// One completed call.
#[derive(Debug, Clone)]
pub struct Completion {
    pub record: DecisionRecord,
    pub usage: Usage,
    pub elapsed: Duration,
    pub attempts: u32,
    /// The seed actually used. Replaying this call with this seed, this
    /// prompt and this model reproduces the episode.
    pub seed: u64,
}

/// The client.
pub struct Agent {
    cfg: AgentConfig,
    http: ureq::Agent,
    /// Fallback when no seed was configured. Fixed at construction so a run
    /// is reproducible from the agent's own record even when the caller did
    /// not think to pin one.
    fallback_seed: u64,
}

impl Agent {
    pub fn new(cfg: AgentConfig) -> Self {
        let http = ureq::Agent::config_builder()
            .timeout_global(Some(cfg.timeout))
            // Hand back non-2xx responses instead of turning them into a bare
            // status code. A 400 whose body says which field the server
            // rejected is a two-minute fix; a 400 with no body is a guessing
            // game, and local servers disagree about which knobs they accept.
            .http_status_as_error(false)
            .build()
            .new_agent();
        let fallback_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| u64::from(d.subsec_nanos()) ^ d.as_secs())
            .unwrap_or(0x5A_4A_17_A4_u64);
        Self {
            cfg,
            http,
            fallback_seed,
        }
    }

    /// The seed this agent will use for a given attempt.
    ///
    /// Derived rather than random per call, so the whole run replays from one
    /// number while no two attempts share a sample path.
    ///
    /// Attempts are 1-based and the first one uses the configured seed
    /// unchanged. Offsetting it would mean the number you pin is never the
    /// number actually used, which defeats the point of pinning it.
    pub fn seed_for(&self, attempt: u32) -> u64 {
        self.cfg
            .seed
            .unwrap_or(self.fallback_seed)
            .wrapping_add(u64::from(attempt.saturating_sub(1)))
    }

    pub fn config(&self) -> &AgentConfig {
        &self.cfg
    }

    /// Ask for a decision.
    ///
    /// `authority` comes from the prompt, not from the model, and is stamped
    /// onto the resulting record.
    pub fn decide(
        &self,
        prompt: &Prompt,
        policy_version: PolicyVersion,
    ) -> Result<Completion, AgentError> {
        let started = std::time::Instant::now();
        let mut last = String::new();

        for attempt in 1..=(self.cfg.max_retries + 1) {
            match self.attempt(prompt, policy_version, attempt) {
                Ok((record, usage)) => {
                    return Ok(Completion {
                        record,
                        usage,
                        elapsed: started.elapsed(),
                        attempts: attempt,
                        seed: self.seed_for(attempt),
                    });
                }
                // A transport or status failure is not the model's fault and
                // retrying will not reword it into working. Fail immediately
                // rather than burning the budget three times over.
                Err(e @ (AgentError::Transport(_) | AgentError::Status { .. })) => return Err(e),
                Err(e) => last = e.to_string(),
            }
        }

        Err(AgentError::Exhausted {
            attempts: self.cfg.max_retries + 1,
            last,
        })
    }

    fn attempt(
        &self,
        prompt: &Prompt,
        policy_version: PolicyVersion,
        attempt: u32,
    ) -> Result<(DecisionRecord, Usage), AgentError> {
        let grammar = match self.cfg.constrain {
            Constrain::Grammar => Some(grammar::DRAFT_DECISION_GBNF.to_string()),
            _ => None,
        };
        let schema = matches!(self.cfg.constrain, Constrain::Schema)
            .then(|| ("draft_decision".to_string(), grammar::draft_decision_schema()));

        let (content, usage) = self.complete(
            &prompt.system(),
            prompt.user(),
            grammar.as_deref(),
            schema,
            self.cfg.temperature + 0.1 * f64::from(attempt - 1),
            self.seed_for(attempt),
        )?;

        let draft: DraftDecision = serde_json::from_str(extract_json(&content))
            .map_err(|e| AgentError::Parse(format!("{e}; got: {content}")))?;

        let record = draft.seal(policy_version, prompt.authority())?;
        Ok((record, usage))
    }

    /// One grammar-constrained completion, returning the raw content and the
    /// token usage.
    ///
    /// The shared transport under both agents in this system: the Warden's
    /// decisions and the Deviant's attacks are the same HTTP call with a
    /// different grammar. Exposed so the adversary can reuse it rather than
    /// carry a second copy of the client, the retry policy, and the four ways
    /// a local server can disagree about a request body.
    ///
    /// `grammar` (GBNF) is preferred; `schema` is the weaker `response_format`
    /// fallback; passing neither lets the model answer freely and leaves the
    /// caller to extract the JSON.
    pub fn complete(
        &self,
        system: &str,
        user: &str,
        grammar: Option<&str>,
        schema: Option<(String, serde_json::Value)>,
        temperature: f64,
        seed: u64,
    ) -> Result<(String, Usage), AgentError> {
        let mut body = serde_json::json!({
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.cfg.max_tokens,
            "repeat_penalty": self.cfg.repeat_penalty,
            // Stream the reply. A buffered (stream:false) response sends nothing
            // until the whole generation is done, so a long answer that outruns a
            // tunnel's response timeout (Cloudflare's is ~100s) is cut off as a
            // 524 even though the model is still working. Streaming flows bytes
            // immediately, so the timeout never fires however long generation
            // runs. include_usage puts the token counts in the final chunk.
            "stream": true,
            "stream_options": {"include_usage": true},
            "seed": seed,
            // llama.cpp: reuse the cached prefix rather than reprocessing it.
            // Unknown keys are ignored by servers that do not implement them.
            "cache_prompt": true,
        });

        if let Some(g) = grammar {
            body["grammar"] = serde_json::Value::String(g.to_string());
        } else if let Some((name, s)) = schema {
            body["response_format"] = serde_json::json!({
                "type": "json_schema",
                "json_schema": { "name": name, "strict": true, "schema": s },
            });
        }

        let url = format!("{}/chat/completions", self.cfg.base_url.trim_end_matches('/'));

        // Retry a *transient* failure — a 529/503 from a busy tunnel or a model
        // still warming, a reset or timed-out connection — with exponential
        // backoff. A non-transient status (400/404 bad request, 401/403 auth) is
        // the server refusing a request it understood, so it fails fast: retrying
        // only repeats it. This is why the reasoning path was losing items to a
        // flaky endpoint while every good response came back fine.
        let mut resp = {
            let mut attempt = 0u32;
            loop {
                attempt += 1;
                let mut req = self.http.post(&url).header("Content-Type", "application/json");
                if !self.cfg.api_key.is_empty() {
                    req = req.header("Authorization", &format!("Bearer {}", self.cfg.api_key));
                }
                match req.send_json(&body) {
                    Ok(r) => {
                        if is_transient_status(r.status().as_u16()) && attempt < MAX_HTTP_ATTEMPTS {
                            std::thread::sleep(retry_backoff(attempt));
                            continue;
                        }
                        break r;
                    }
                    Err(e) => {
                        // A transport error (reset, timeout) is transient too.
                        if attempt < MAX_HTTP_ATTEMPTS {
                            std::thread::sleep(retry_backoff(attempt));
                            continue;
                        }
                        return Err(AgentError::Transport(e.to_string()));
                    }
                }
            }
        };

        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let detail = resp
                .body_mut()
                .read_to_string()
                .unwrap_or_else(|e| format!("(body unreadable: {e})"));
            return Err(AgentError::Status {
                status,
                body: detail.chars().take(2000).collect(),
            });
        }

        // An empty reply is not a transport error: a thinking model can spend its
        // whole token budget and never reach a final answer. Return it and let the
        // caller grade it as a non-answer (a MISS), so a truncated item is not
        // mistaken for a server failure. SAMARITAN_STREAM=1 echoes tokens live.
        let live = std::env::var("SAMARITAN_STREAM")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        let reader = std::io::BufReader::new(resp.body_mut().as_reader());
        let (content, usage) = accumulate_sse(reader, live)?;
        Ok((content, usage))
    }
}

/// Max attempts for one completion when the failure looks transient (overload,
/// gateway, reset). Enough to ride out a burst of 529s from a busy tunnel or a
/// model still loading, without stalling a run for minutes.
const MAX_HTTP_ATTEMPTS: u32 = 5;

/// Transient HTTP statuses worth retrying: overload, rate-limit, gateway, timeout.
/// Everything else the server understood and refused, so it should not be retried.
fn is_transient_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 522 | 524 | 529)
}

/// Exponential backoff before retry `attempt` (1-based): 0.5s, 1s, 2s, 4s, 8s cap.
fn retry_backoff(attempt: u32) -> std::time::Duration {
    let secs = 0.5_f64 * 2_f64.powi((attempt.saturating_sub(1)) as i32);
    std::time::Duration::from_millis((secs.min(8.0) * 1000.0) as u64)
}

/// Assemble an OpenAI-style SSE stream into the full content and token usage.
///
/// Reads `data: {chunk}` lines, appends each `choices[0].delta.content`, stops at
/// `data: [DONE]` (or EOF), and takes the `usage` object from whichever chunk
/// carries it — the final one, sent because we ask for `include_usage`. Lines
/// that are not `data:` (SSE comments / keep-alives) and any chunk that does not
/// parse are skipped, so a stray keep-alive never derails a reply. Kept pure and
/// reader-generic so it is unit-testable without a live server.
fn accumulate_sse<R: std::io::BufRead>(reader: R, live: bool) -> Result<(String, Usage), AgentError> {
    use std::io::Write as _;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage = Usage { prompt_tokens: 0, completion_tokens: 0, cached_tokens: 0 };
    let echo = |tok: &str| {
        if live {
            print!("{tok}");
            let _ = std::io::stdout().flush();
        }
    };
    for line in reader.lines() {
        let line = line.map_err(|e| AgentError::Transport(e.to_string()))?;
        let data = match line.trim().strip_prefix("data:") {
            Some(d) => d.trim(),
            None => continue,
        };
        if data == "[DONE]" {
            break;
        }
        let v: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let delta = &v["choices"][0]["delta"];
        if let Some(tok) =
            delta["content"].as_str().or_else(|| v["choices"][0]["message"]["content"].as_str())
        {
            content.push_str(tok);
            echo(tok);
        }
        // A thinking model's reasoning is often streamed in a separate field
        // (reasoning_content, or reasoning) with only the final answer in content.
        // Capture it so the trace is not lost and a reply that is all-reasoning
        // (a truncated, answer-less item) does not look empty.
        if let Some(tok) =
            delta["reasoning_content"].as_str().or_else(|| delta["reasoning"].as_str())
        {
            reasoning.push_str(tok);
            echo(tok);
        }
        if !v["usage"].is_null() {
            usage = Usage {
                prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(usage.prompt_tokens),
                completion_tokens: v["usage"]["completion_tokens"]
                    .as_u64()
                    .unwrap_or(usage.completion_tokens),
                cached_tokens: v["usage"]["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(usage.cached_tokens),
            };
        }
    }
    if live {
        println!();
    }
    // Fold separated reasoning back into a <think> block, so the answer-extractor
    // sees the shape it already handles and nothing is lost. If the server inlined
    // its thinking into content, reasoning is empty and content passes through.
    let full = match (reasoning.is_empty(), content.is_empty()) {
        (false, false) => format!("<think>{reasoning}</think>\n{content}"),
        (false, true) => format!("<think>{reasoning}</think>"),
        (true, _) => content,
    };
    Ok((full, usage))
}

/// Pull the JSON object out of a possibly-wrapped reply. Public so other
/// crates reusing [`Agent::complete`] recover the object the same way.
pub fn extract_json_object(content: &str) -> &str {
    extract_json(content)
}

/// Pull the JSON object out of a reply that may be wrapped.
///
/// With a grammar attached the content is already bare JSON and this is the
/// identity function. Without one, reasoning-tuned models routinely wrap the
/// answer in a fence or precede it with a chain of thought, and recovering
/// the object is much cheaper than a second 55-second generation.
fn extract_json(content: &str) -> &str {
    let s = content.trim();

    if let Some(rest) = s.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim();
        }
    }

    // Outermost balanced braces, ignoring anything inside a string literal so
    // that a `}` in a file path or a patch body does not truncate the object.
    let bytes = s.as_bytes();
    let Some(start) = bytes.iter().position(|b| *b == b'{') else {
        return s;
    };
    let (mut depth, mut in_str, mut escaped) = (0i32, false, false);
    for (i, b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &s[start..=i];
                }
            }
            _ => {}
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_bare_json_unchanged() {
        assert_eq!(extract_json(r#"{"a":1}"#), r#"{"a":1}"#);
    }

    #[test]
    fn strips_a_code_fence() {
        assert_eq!(extract_json("```json\n{\"a\":1}\n```"), r#"{"a":1}"#);
        assert_eq!(extract_json("```\n{\"a\":1}\n```"), r#"{"a":1}"#);
    }

    #[test]
    fn skips_preamble_from_a_reasoning_model() {
        let s = "Let me think about this.\nOkay, here it is:\n{\"a\":1}";
        assert_eq!(extract_json(s), r#"{"a":1}"#);
    }

    #[test]
    fn a_brace_inside_a_string_does_not_end_the_object() {
        // The case that matters: a patch body or a path containing braces.
        let s = r#"{"contents":"fn main() { let x = 1; }","b":2}"#;
        assert_eq!(extract_json(s), s);
    }

    #[test]
    fn an_escaped_quote_does_not_end_the_string() {
        let s = r#"{"a":"he said \"}\" loudly","b":2}"#;
        assert_eq!(extract_json(s), s);
    }

    #[test]
    fn nested_objects_are_kept_whole() {
        let s = r#"{"a":{"b":{"c":1}}}"#;
        assert_eq!(extract_json(s), s);
    }

    #[test]
    fn sse_stream_assembles_content_and_usage() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n",
            "\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n",
            "data: [DONE]\n",
        );
        let (content, usage) = accumulate_sse(std::io::Cursor::new(stream), false).unwrap();
        assert_eq!(content, "Hello");
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 2);
    }

    #[test]
    fn sse_folds_separated_reasoning_into_a_think_block() {
        // A server that streams the trace in reasoning_content and only the answer
        // in content — the reasoning is folded back so nothing is lost.
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"6*7 is 42\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Answer: 42\"}}]}\n",
            "data: [DONE]\n",
        );
        let (full, _) = accumulate_sse(std::io::Cursor::new(stream), false).unwrap();
        assert_eq!(full, "<think>6*7 is 42</think>\nAnswer: 42");
    }

    #[test]
    fn sse_reasoning_only_reply_keeps_its_trace_and_is_not_empty() {
        // The truncated-thinking case: all reasoning, no final answer. It must come
        // back as its trace (to be graded a MISS), not as an empty transport error.
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"still thinking\"}}]}\n",
            "data: [DONE]\n",
        );
        let (full, _) = accumulate_sse(std::io::Cursor::new(stream), false).unwrap();
        assert_eq!(full, "<think>still thinking</think>");
    }

    #[test]
    fn sse_skips_keepalives_and_unparsable_lines() {
        // An SSE comment, a garbled chunk, and a usage-only tail must not derail it.
        let stream = concat!(
            ": keep-alive\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n",
            "data: not-json\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n",
            "data: [DONE]\n",
        );
        let (content, _) = accumulate_sse(std::io::Cursor::new(stream), false).unwrap();
        assert_eq!(content, "AB");
    }

    #[test]
    fn transient_statuses_retry_but_refusals_do_not() {
        assert!(is_transient_status(529));
        assert!(is_transient_status(524));
        assert!(is_transient_status(503));
        assert!(!is_transient_status(200));
        assert!(!is_transient_status(400));
        assert!(!is_transient_status(404));
        // Backoff grows with the attempt and stays capped.
        assert!(retry_backoff(1) < retry_backoff(3));
        assert!(retry_backoff(10) <= std::time::Duration::from_secs(8));
    }
}

#[cfg(test)]
mod seed_tests {
    use super::*;

    #[test]
    fn the_first_attempt_uses_the_seed_you_pinned() {
        // Otherwise the number in the config is never the number used, and a
        // "reproducible" run reproduces something you did not ask for.
        let a = Agent::new(AgentConfig {
            seed: Some(20260911),
            ..Default::default()
        });
        assert_eq!(a.seed_for(1), 20260911);
    }

    #[test]
    fn retries_walk_a_different_sample_path() {
        let a = Agent::new(AgentConfig {
            seed: Some(100),
            ..Default::default()
        });
        assert_eq!((a.seed_for(1), a.seed_for(2), a.seed_for(3)), (100, 101, 102));
    }

    #[test]
    fn an_unpinned_run_still_records_one_stable_seed() {
        // No configured seed still means a knowable run: the agent fixes one
        // at construction rather than drawing per call.
        let a = Agent::new(AgentConfig::default());
        assert_eq!(a.seed_for(1), a.seed_for(1));
    }
}
