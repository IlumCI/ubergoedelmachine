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
    pub fn seed_for(&self, attempt: u32) -> u64 {
        self.cfg
            .seed
            .unwrap_or(self.fallback_seed)
            .wrapping_add(u64::from(attempt))
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
        let mut body = serde_json::json!({
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": prompt.system()},
                {"role": "user", "content": prompt.user()},
            ],
            // A later retry is not a chance to say the same thing again, so
            // it samples a little wider. Only relevant without a grammar.
            "temperature": self.cfg.temperature + 0.1 * f64::from(attempt - 1),
            "max_tokens": self.cfg.max_tokens,
            "stream": false,
            "seed": self.seed_for(attempt),
            // llama.cpp: reuse the cached prefix rather than reprocessing it.
            // Unknown keys are ignored by servers that do not implement them.
            "cache_prompt": true,
        });

        match self.cfg.constrain {
            Constrain::Grammar => {
                body["grammar"] = serde_json::Value::String(
                    grammar::DRAFT_DECISION_GBNF.to_string(),
                );
            }
            Constrain::Schema => {
                body["response_format"] = serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "draft_decision",
                        "strict": true,
                        "schema": grammar::draft_decision_schema(),
                    }
                });
            }
            Constrain::None => {}
        }

        let url = format!("{}/chat/completions", self.cfg.base_url.trim_end_matches('/'));
        let mut req = self.http.post(&url).header("Content-Type", "application/json");
        if !self.cfg.api_key.is_empty() {
            req = req.header("Authorization", &format!("Bearer {}", self.cfg.api_key));
        }

        let mut resp = match req.send_json(&body) {
            Ok(r) => r,
            Err(ureq::Error::StatusCode(code)) => {
                return Err(AgentError::Status {
                    status: code,
                    body: String::from("(status error)"),
                });
            }
            Err(e) => return Err(AgentError::Transport(e.to_string())),
        };

        let v: serde_json::Value = resp
            .body_mut()
            .read_json()
            .map_err(|e| AgentError::Shape(e.to_string()))?;

        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| AgentError::Shape(format!("no message content in {v}")))?;

        let usage = Usage {
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
            cached_tokens: v["usage"]["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
        };

        let draft: DraftDecision = serde_json::from_str(extract_json(content))
            .map_err(|e| AgentError::Parse(format!("{e}; got: {content}")))?;

        let record = draft.seal(policy_version, prompt.authority())?;
        Ok((record, usage))
    }
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
}
