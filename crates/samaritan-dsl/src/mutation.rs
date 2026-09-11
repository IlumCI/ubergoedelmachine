//! The mutation grammar, and the policy the nested search adapts over it.
//!
//! Two rules here are load-bearing for the levels that do not exist yet, and
//! both are cheap now and expensive to retrofit:
//!
//! 1. **Mutations are data, not code.** [`Mutation`] is an enum interpreted by
//!    an applier, never a closure. Level 2 searches over the *grammar itself*,
//!    which it can only do if mutations are values it can inspect and rewrite.
//! 2. **[`MutationPolicy`] has value semantics and no interior mutability.**
//!    NRPA passes the policy by value into the recursive call so that
//!    adaptations made inside a deeper level vanish when it returns. That is
//!    where speculative self-modification gets its free rollback. An
//!    `Arc<Mutex<_>>` slipped in for convenience would silently destroy the
//!    isolation that makes the higher levels safe.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::DslError;

/// Identity of a lesson in the mutable lesson memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LessonId(pub u64);

/// A tunable in the mutable half of the router policy.
///
/// These name *proposals* about autonomy. The frozen kernel clamps every one of
/// them against a human-set ceiling that no mutation can name, let alone raise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Knob {
    /// Consecutive clean approvals before an action class may be promoted.
    PromotionStreak,
    /// Brier score above which no promotion is allowed at all.
    CalibrationCeiling,
    /// How many lessons are injected into a decision prompt.
    LessonBudget,
    /// Sampling temperature for the decision model.
    DecisionTemperature,
}

/// One edit to the mutable state.
///
/// The last two variants exist so the grammar does not have to change when
/// levels 2 and 3 arrive. The applier rejects them today.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Mutation {
    /// Level 1: add a lesson mined from the ledger.
    LessonAdd { text: String },
    /// Level 1: retire a lesson that stopped paying for itself.
    LessonRemove { id: LessonId },
    /// Level 1: change how strongly a lesson is weighted for inclusion.
    LessonReweight { id: LessonId, delta: f64 },
    /// Level 1: move a router knob, subject to the frozen ceiling.
    ThresholdSet { knob: Knob, value: f64 },
    /// Level 2: rewrite the grammar the search samples from. Not yet enabled.
    GrammarEdit { spec: serde_json::Value },
    /// Level 3: patch the search code itself. Not yet enabled.
    CodePatch { unified_diff: String },
}

/// Stable key identifying a *class* of mutation.
///
/// The NRPA policy is a distribution over these, not over concrete mutations —
/// there are unboundedly many possible lesson texts but only a handful of kinds
/// of move, and the policy has to generalise across episodes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MutationCode(pub String);

impl std::fmt::Display for MutationCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Mutation {
    /// The policy key for this mutation.
    pub fn code(&self) -> MutationCode {
        let s = match self {
            Mutation::LessonAdd { .. } => "lesson.add".to_string(),
            Mutation::LessonRemove { .. } => "lesson.remove".to_string(),
            Mutation::LessonReweight { .. } => "lesson.reweight".to_string(),
            Mutation::ThresholdSet { knob, .. } => {
                format!("threshold.{}", serde_plain_knob(*knob))
            }
            Mutation::GrammarEdit { .. } => "grammar.edit".to_string(),
            Mutation::CodePatch { .. } => "code.patch".to_string(),
        };
        MutationCode(s)
    }

    /// The self-reference level a mutation belongs to.
    ///
    /// Used both to route it to the right search level and to pick the error
    /// budget slice its certificate spends.
    pub fn level(&self) -> u8 {
        match self {
            Mutation::LessonAdd { .. }
            | Mutation::LessonRemove { .. }
            | Mutation::LessonReweight { .. }
            | Mutation::ThresholdSet { .. } => 1,
            Mutation::GrammarEdit { .. } => 2,
            Mutation::CodePatch { .. } => 3,
        }
    }
}

fn serde_plain_knob(k: Knob) -> &'static str {
    match k {
        Knob::PromotionStreak => "promotion_streak",
        Knob::CalibrationCeiling => "calibration_ceiling",
        Knob::LessonBudget => "lesson_budget",
        Knob::DecisionTemperature => "decision_temperature",
    }
}

/// The NRPA policy: log-weights over mutation classes.
///
/// Plain owned data. Cloning gives a fully independent copy, which is exactly
/// the property the nested search relies on.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MutationPolicy {
    weights: BTreeMap<MutationCode, f64>,
}

impl MutationPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Log-weight for a class; unseen classes start at zero, i.e. uniform.
    pub fn weight(&self, code: &MutationCode) -> f64 {
        self.weights.get(code).copied().unwrap_or(0.0)
    }

    pub fn set_weight(&mut self, code: MutationCode, w: f64) -> Result<(), DslError> {
        if !w.is_finite() {
            return Err(DslError::BadWeight(w));
        }
        self.weights.insert(code, w);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&MutationCode, f64)> {
        self.weights.iter().map(|(k, v)| (k, *v))
    }

    /// Softmax over `legal` under the current weights, numerically stabilised.
    ///
    /// Returns probabilities aligned with `legal`. An empty `legal` yields an
    /// empty vector rather than an error; callers treat that as a dead end.
    pub fn distribution(&self, legal: &[MutationCode]) -> Vec<f64> {
        if legal.is_empty() {
            return Vec::new();
        }
        let ws: Vec<f64> = legal.iter().map(|c| self.weight(c)).collect();
        let max = ws.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = ws.iter().map(|w| (w - max).exp()).collect();
        let z: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / z).collect()
    }

    /// The NRPA adapt step (Rosin 2011), as a pure function.
    ///
    /// For each step of the best sequence, push weight toward the move that was
    /// actually taken and pull it away from every move that was legal at that
    /// point, in proportion to how likely the *current* policy already was to
    /// pick each one. Gradients are computed against `self` throughout while
    /// updates accumulate into the copy, which is what keeps a single adapt
    /// call from chasing its own tail.
    ///
    /// `trace` is the best sequence as (chosen, legal-at-that-step) pairs.
    pub fn adapt(&self, trace: &[(MutationCode, Vec<MutationCode>)], alpha: f64) -> Self {
        let mut next = self.clone();
        for (chosen, legal) in trace {
            let probs = self.distribution(legal);
            *next.weights.entry(chosen.clone()).or_insert(0.0) += alpha;
            for (code, p) in legal.iter().zip(probs) {
                *next.weights.entry(code.clone()).or_insert(0.0) -= alpha * p;
            }
        }
        next
    }
}
