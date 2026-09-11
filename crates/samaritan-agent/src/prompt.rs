//! Assembling prompts so the server can skip most of the work.
//!
//! At roughly nine tokens per second, prompt processing is not a rounding
//! error — it is most of the wall clock. The single largest speed lever
//! available is therefore not sampling settings but **prefix reuse**:
//! llama.cpp caches the KV state of a prompt prefix per slot, and a request
//! whose prefix matches an earlier one skips straight to the first differing
//! token.
//!
//! That turns prompt layout into a performance decision, so it is encoded in
//! the type rather than left to whoever writes the next call site. A
//! [`Prompt`] is built in three bands, most stable first:
//!
//! 1. **Constant** — the instructions. Identical for the lifetime of a run.
//! 2. **Policy** — lessons and knobs. Identical until a mutation commits,
//!    which is once per level-1 round rather than once per episode.
//! 3. **Episode** — the task, the working state, observations. Different
//!    every time.
//!
//! Bands 1 and 2 are the same across every playout in a round, so they are
//! processed once and reused for the rest. Interleaving a single
//! episode-specific token into band 1 — a timestamp, an episode id, a
//! shuffled lesson order — invalidates the cache for everything after it and
//! silently costs more than every other optimisation in this crate returns.

use samaritan_dsl::Authority;

/// A prompt laid out for maximum prefix reuse.
pub struct Prompt {
    constant: String,
    policy: String,
    episode: String,
    /// The least-trusted source that fed into the episode band.
    authority: Authority,
}

/// The instructions. Deliberately a `const`: if this is ever built by
/// formatting, it stops being stable and the cache stops hitting.
pub const SYSTEM: &str = r#"You are the reasoning component of an automated software-repair harness.

You do not act. You emit one JSON object describing a decision, and a separate
component decides whether any of it is allowed to happen. Prose outside the
JSON object is discarded.

Rules for the object you emit:

- "options" must contain at least two genuinely different approaches. One of
  them may be to do nothing. A single-option decision is a rationalisation and
  is rejected by the parser.
- "chosen" is the index of the option you are taking.
- "prediction.outcome" must be something the test suite can settle, and
  "prediction.confidence" is your honest probability that it happens. You are
  scored on calibration: claiming 0.95 and being wrong costs more than
  claiming 0.6 and being wrong. Do not inflate it.
- Every action must describe itself accurately. "kind" is what the action
  does, "reversibility" is how hard it is to undo, "blast_radius" is how far
  the consequences reach. Understating any of these is treated as a violation,
  not as an error.
- Paths are relative to the repository root. You cannot reach outside it.

You are working from a failing test suite. The tests are correct; the source
is wrong. Change the source, not the tests."#;

impl Prompt {
    pub fn new() -> Self {
        Self {
            constant: SYSTEM.to_string(),
            policy: String::new(),
            episode: String::new(),
            authority: Authority::Task,
        }
    }

    /// Band 2: lessons and configuration, stable within a round.
    ///
    /// Lessons come from the agent's own history, so anything built here is
    /// [`Authority::Agent`] at best — never `Task`.
    pub fn with_policy(mut self, lessons: &[String]) -> Self {
        if lessons.is_empty() {
            return self;
        }
        let mut s = String::from("\n\nLessons from previous episodes:\n");
        for (i, l) in lessons.iter().enumerate() {
            s.push_str(&format!("{}. {}\n", i + 1, l.trim()));
        }
        self.policy = s;
        self.authority = self.authority.least(Authority::Agent);
        self
    }

    /// Band 3, the task itself. From the corpus, so it does not taint.
    pub fn with_task(mut self, prompt: &str, failing: &str) -> Self {
        self.episode.push_str(&format!(
            "\n\nTask: {}\n\nThe suite currently fails:\n{}\n",
            prompt.trim(),
            failing.trim()
        ));
        self
    }

    /// Band 3, content read out of the environment.
    ///
    /// This is the call that taints the decision. File contents, stdout and
    /// error text are all writable by an adversary sharing the arena, so
    /// anything downstream of this can never run unattended — see
    /// [`Authority::Observed`] and `samaritan_kernel::route`.
    pub fn with_observation(mut self, label: &str, content: &str) -> Self {
        self.episode
            .push_str(&format!("\n\n{}:\n{}\n", label.trim(), content.trim()));
        self.authority = self.authority.least(Authority::Observed);
        self
    }

    /// The provenance the resulting decision must carry.
    ///
    /// Computed from what was actually put into the prompt rather than
    /// reported by the model, which is the whole point: a model asked to
    /// declare its own trust level would declare itself trusted.
    pub fn authority(&self) -> Authority {
        self.authority
    }

    /// The stable portion, sent as the system message so the server can cache
    /// its KV state across every request in the round.
    pub fn system(&self) -> String {
        format!("{}{}", self.constant, self.policy)
    }

    /// The varying portion, sent as the user message.
    pub fn user(&self) -> &str {
        &self.episode
    }

    /// How much of this prompt is expected to hit a warm cache, as a fraction
    /// of characters. Reported to the ledger so a regression in prompt layout
    /// shows up as a throughput drop with a cause attached, rather than as a
    /// mysteriously slower run.
    pub fn cacheable_fraction(&self) -> f64 {
        let stable = (self.constant.len() + self.policy.len()) as f64;
        let total = stable + self.episode.len() as f64;
        if total == 0.0 { 0.0 } else { stable / total }
    }
}

impl Default for Prompt {
    fn default() -> Self {
        Self::new()
    }
}
