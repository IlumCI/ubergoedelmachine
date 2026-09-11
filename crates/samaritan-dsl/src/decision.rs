//! The decision language.
//!
//! The model does not act. It emits a [`DecisionRecord`] describing what it
//! believes and what it wants done, and something else decides whether any of
//! that gets to happen. Prose is not executable; only records are.
//!
//! Every type here is constructed through a checked path, and `serde` is wired
//! through the same path via `try_from`, so a record that deserializes is a
//! record that holds its invariants.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DslError;

/// Identity of a single decision, stable across the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DecisionId(pub Uuid);

impl DecisionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for DecisionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DecisionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What kind of access an action needs.
///
/// Ordered loosely by how much trouble it can cause, but the router derives
/// tiers from the whole triple rather than from this alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// Reads state without changing it.
    Read,
    /// Writes to the working tree.
    Write,
    /// Runs a subprocess.
    Exec,
    /// Touches the network.
    Net,
    /// Rewrites git history, which destroys information rather than adding it.
    GitHistory,
}

/// How hard it would be to undo the action after the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reversibility {
    /// Undo costs nothing.
    Trivial,
    /// Undo means restoring the episode snapshot.
    Snapshot,
    /// Undoable, but the undo is itself real work.
    Costly,
    /// Cannot be undone. Nothing here is ever automatic.
    Irreversible,
}

/// How far the consequences reach if the action goes wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlastRadius {
    /// Confined to the sandbox worktree for one episode.
    Episode,
    /// Reaches the real repository.
    Repo,
    /// Reaches the host machine.
    Machine,
    /// Leaves the machine. Visible to other people.
    External,
}

/// One thing the model wants done.
///
/// Note what is absent: there is no tier field. The model reports facts about
/// its action and the frozen router derives the tier from them. Letting the
/// proposer grade its own proposal would make the whole gate decorative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposedAction {
    pub kind: ActionKind,
    pub reversibility: Reversibility,
    pub blast_radius: BlastRadius,
    /// Human-readable description of the concrete operation.
    pub intent: String,
    /// The operation itself, interpreted by `samaritan-exec`.
    pub payload: serde_json::Value,
}

/// A probability, checked on the way in.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Confidence(f64);

impl Confidence {
    pub fn new(v: f64) -> Result<Self, DslError> {
        if !v.is_finite() {
            return Err(DslError::ConfidenceNotFinite(v));
        }
        if !(0.0..=1.0).contains(&v) {
            return Err(DslError::ConfidenceOutOfRange(v));
        }
        Ok(Self(v))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Confidence {
    type Error = DslError;
    fn try_from(v: f64) -> Result<Self, Self::Error> {
        Self::new(v)
    }
}

impl From<Confidence> for f64 {
    fn from(c: Confidence) -> f64 {
        c.0
    }
}

/// What the model expects to happen, stated before it finds out.
///
/// This is the raw material for calibration scoring: `confidence` is compared
/// against the recorded outcome, so the model is on the record every time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    /// The expected observable result, in terms the outcome check can settle.
    pub outcome: String,
    pub confidence: Confidence,
}

/// One option that was weighed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionOption {
    pub summary: String,
    /// Why this option was taken, or was not.
    pub assessment: String,
}

/// Where the content that shaped a decision came from.
///
/// Adapted from Conway Automaton's policy engine, which derives an authority
/// level from a tool call's input source so that untrusted content cannot
/// authorise as much as a direct instruction. The idea earns its place here
/// for a sharper reason than prompt injection in general: in the arena, the
/// Deviant writes files, emits test output, and produces error messages, and
/// the Warden reads all three. Without provenance, an adversary that cannot
/// break the router directly can simply *write* the reasoning that persuades
/// the Warden to act.
///
/// Ordered by how little the source is trusted, so combining two inputs is a
/// `max` and the least trustworthy one wins — the same direction as `Tier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Authority {
    /// The task statement, from the corpus. Not agent-writable.
    Task,
    /// The agent's own prior reasoning and its lesson memory.
    Agent,
    /// Observed from the environment: file contents, stdout, stderr, exit
    /// messages. In the arena this is adversary-writable, so a decision
    /// resting on it can never run unattended.
    Observed,
}

impl Authority {
    /// The lower of two authorities, which is to say the less trusted.
    pub fn least(self, other: Authority) -> Authority {
        self.max(other)
    }

    /// The least-trusted source among several. An empty iterator yields
    /// [`Authority::Task`]: a decision influenced by nothing observed is not
    /// tainted by anything.
    pub fn least_of(sources: impl IntoIterator<Item = Authority>) -> Authority {
        sources.into_iter().fold(Authority::Task, Authority::least)
    }
}

/// Hash of the mutable policy state that produced a decision.
///
/// Ties every decision back to the exact configuration that generated it, so
/// the search can attribute outcomes to the mutation that caused them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PolicyVersion(pub crate::hash::Digest);

/// The unit of thought.
///
/// Fields are private and the only constructor validates, so holding one of
/// these is proof that it is well-formed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "DecisionRecordRaw")]
pub struct DecisionRecord {
    id: DecisionId,
    situation: String,
    options: Vec<DecisionOption>,
    chosen: usize,
    rationale: String,
    prediction: Prediction,
    actions: Vec<ProposedAction>,
    policy_version: PolicyVersion,
    authority: Authority,
}

/// Deserialization shim. Everything arriving from outside lands here first and
/// is forced through [`DecisionRecord::new`].
#[derive(Deserialize)]
struct DecisionRecordRaw {
    id: DecisionId,
    situation: String,
    options: Vec<DecisionOption>,
    chosen: usize,
    rationale: String,
    prediction: Prediction,
    actions: Vec<ProposedAction>,
    policy_version: PolicyVersion,
    /// Absent means untainted. A model that omits it does not thereby get
    /// more authority than one that admits to reading a file.
    #[serde(default = "default_authority")]
    authority: Authority,
}

fn default_authority() -> Authority {
    Authority::Task
}

impl TryFrom<DecisionRecordRaw> for DecisionRecord {
    type Error = DslError;
    fn try_from(r: DecisionRecordRaw) -> Result<Self, Self::Error> {
        Self::new(
            r.id,
            r.situation,
            r.options,
            r.chosen,
            r.rationale,
            r.prediction,
            r.actions,
            r.policy_version,
            r.authority,
        )
    }
}

fn non_blank(s: &str, field: &'static str) -> Result<(), DslError> {
    if s.trim().is_empty() {
        return Err(DslError::Blank { field });
    }
    Ok(())
}

impl DecisionRecord {
    /// The single checked constructor.
    ///
    /// Requiring two options is not pedantry. A record with one option is a
    /// rationalisation, and the search cannot learn anything from a policy that
    /// never considered doing otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: DecisionId,
        situation: String,
        options: Vec<DecisionOption>,
        chosen: usize,
        rationale: String,
        prediction: Prediction,
        actions: Vec<ProposedAction>,
        policy_version: PolicyVersion,
        authority: Authority,
    ) -> Result<Self, DslError> {
        if options.len() < 2 {
            return Err(DslError::TooFewOptions(options.len()));
        }
        if chosen >= options.len() {
            return Err(DslError::ChosenOutOfRange {
                chosen,
                count: options.len(),
            });
        }
        non_blank(&situation, "situation")?;
        non_blank(&rationale, "rationale")?;
        non_blank(&prediction.outcome, "prediction.outcome")?;
        for o in &options {
            non_blank(&o.summary, "option.summary")?;
        }
        for a in &actions {
            non_blank(&a.intent, "action.intent")?;
        }

        Ok(Self {
            id,
            situation,
            options,
            chosen,
            rationale,
            prediction,
            actions,
            policy_version,
            authority,
        })
    }

    pub fn id(&self) -> DecisionId {
        self.id
    }
    pub fn situation(&self) -> &str {
        &self.situation
    }
    pub fn options(&self) -> &[DecisionOption] {
        &self.options
    }
    pub fn chosen_index(&self) -> usize {
        self.chosen
    }
    /// Infallible: the constructor already proved the index is in range.
    pub fn chosen(&self) -> &DecisionOption {
        &self.options[self.chosen]
    }
    pub fn rationale(&self) -> &str {
        &self.rationale
    }
    pub fn prediction(&self) -> &Prediction {
        &self.prediction
    }
    pub fn actions(&self) -> &[ProposedAction] {
        &self.actions
    }
    pub fn policy_version(&self) -> PolicyVersion {
        self.policy_version
    }
    /// The least-trusted source that shaped this decision.
    pub fn authority(&self) -> Authority {
        self.authority
    }
}
