//! The mutable half of the machine: what a self-modification may change.
//!
//! Everything here is searchable. Everything the search must not reach lives
//! in `samaritan-kernel`, which exposes no mutation API, so the partition is
//! enforced by the type system rather than by remembering to check.

use std::collections::{BTreeMap, BTreeSet};

use samaritan_dsl::{Digest, Knob, LessonId, Mutation, hash_json};
use samaritan_kernel::{Admission, Refusal};
use serde::{Deserialize, Serialize};

/// One thing the agent learned, and how strongly it is weighted for
/// inclusion in a prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lesson {
    pub id: LessonId,
    pub text: String,
    pub weight: f64,
}

/// The mutation kinds a level-2 grammar edit is allowed to switch on.
///
/// **This whitelist is the level barrier.** Level 2 may reshape the vocabulary
/// the search draws from, but it may not reach *up* a level: `code_patch` is
/// deliberately absent, so no sequence of grammar edits can enable level-3
/// moves through the back door. A machine that can rewrite the rules of its own
/// move generation must still not be able to promote itself.
const GRAMMAR_ENABLEABLE: &[&str] = &[
    "lesson_add",
    "lesson_remove",
    "lesson_reweight",
    "threshold_set",
];

/// The grammar the search draws its moves from — data, so level 2 can edit it.
///
/// At level 1 the vocabulary is fixed and the search only *combines* moves. The
/// Gödelian step is that this description of what moves *exist* is itself part
/// of the searchable state, so a level-2 rollout can widen, narrow, or retune
/// the move generator before a level-1 search runs inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grammar {
    /// Mutation kinds admitted into the search vocabulary.
    enabled_kinds: BTreeSet<String>,
    /// Step size a lesson-reweight move uses.
    reweight_delta: f64,
    /// Mutations per rollout — the depth half of the budget split.
    depth: usize,
    /// NRPA adapt rate the level below runs with.
    adapt_alpha: f64,
}

impl Default for Grammar {
    fn default() -> Self {
        Self {
            enabled_kinds: GRAMMAR_ENABLEABLE.iter().map(|s| s.to_string()).collect(),
            reweight_delta: 0.5,
            depth: 4,
            adapt_alpha: 1.0,
        }
    }
}

impl Grammar {
    pub fn kind_enabled(&self, kind: &str) -> bool {
        self.enabled_kinds.contains(kind)
    }
    pub fn enabled_kinds(&self) -> impl Iterator<Item = &str> {
        self.enabled_kinds.iter().map(|s| s.as_str())
    }
    pub fn reweight_delta(&self) -> f64 {
        self.reweight_delta
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn adapt_alpha(&self) -> f64 {
        self.adapt_alpha
    }

    /// Apply one grammar operation, or say why it is refused.
    ///
    /// Every refusal here is an invariant the search must not be able to break
    /// from the inside: it cannot promote itself a level, cannot empty its own
    /// vocabulary into a no-op, and cannot set a parameter to a value that is
    /// not a number.
    fn apply_op(&mut self, op: GrammarOp) -> Result<(), String> {
        match op {
            GrammarOp::EnableKind { kind } => {
                if !GRAMMAR_ENABLEABLE.contains(&kind.as_str()) {
                    return Err(format!(
                        "{kind} is not a grammar-enableable kind; level 2 cannot promote itself"
                    ));
                }
                self.enabled_kinds.insert(kind);
            }
            GrammarOp::DisableKind { kind } => {
                if self.enabled_kinds.len() <= 1 && self.enabled_kinds.contains(&kind) {
                    return Err("refusing to empty the grammar: a search with no moves is a \
                                no-op that would score as 'no regression'"
                        .into());
                }
                self.enabled_kinds.remove(&kind);
            }
            GrammarOp::SetReweightDelta { value } => {
                if !value.is_finite() || value <= 0.0 || value > 5.0 {
                    return Err(format!("reweight delta {value} is outside (0, 5]"));
                }
                self.reweight_delta = value;
            }
            GrammarOp::SetDepth { value } => {
                if value == 0 || value > 64 {
                    return Err(format!("rollout depth {value} is outside [1, 64]"));
                }
                self.depth = value;
            }
            GrammarOp::SetAdaptAlpha { value } => {
                if !value.is_finite() || value <= 0.0 || value > 10.0 {
                    return Err(format!("adapt alpha {value} is outside (0, 10]"));
                }
                self.adapt_alpha = value;
            }
        }
        Ok(())
    }
}

/// One edit to the grammar. The typed reading of a
/// [`Mutation::GrammarEdit`]'s free-form spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum GrammarOp {
    EnableKind { kind: String },
    DisableKind { kind: String },
    SetReweightDelta { value: f64 },
    SetDepth { value: usize },
    SetAdaptAlpha { value: f64 },
}

/// The searchable state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyState {
    lessons: Vec<Lesson>,
    knobs: BTreeMap<Knob, f64>,
    next_id: u64,
    /// The move generator's own description. Level 1 reads it; level 2 edits it.
    #[serde(default)]
    grammar: Grammar,
}

impl Default for PolicyState {
    fn default() -> Self {
        let mut knobs = BTreeMap::new();
        knobs.insert(Knob::PromotionStreak, 5.0);
        knobs.insert(Knob::CalibrationCeiling, 0.2);
        knobs.insert(Knob::LessonBudget, 8.0);
        knobs.insert(Knob::DecisionTemperature, 0.7);
        Self {
            lessons: Vec::new(),
            knobs,
            next_id: 1,
            grammar: Grammar::default(),
        }
    }
}

impl PolicyState {
    pub fn lessons(&self) -> &[Lesson] {
        &self.lessons
    }

    /// The grammar this state's search draws from.
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    pub fn knob(&self, k: Knob) -> f64 {
        self.knobs.get(&k).copied().unwrap_or(0.0)
    }

    /// Identity of this exact configuration.
    ///
    /// Stamped onto every decision so an outcome can be attributed to the
    /// state that produced it; without that link the search would be adapting
    /// toward results it cannot trace.
    pub fn digest(&self) -> Digest {
        hash_json(self)
    }

    /// The lessons a prompt should carry, highest weight first and cut to the
    /// budget.
    pub fn prompt_lessons(&self) -> Vec<String> {
        let budget = self.knob(Knob::LessonBudget).max(0.0) as usize;
        let mut v: Vec<&Lesson> = self.lessons.iter().collect();
        // Ties broken by id so the order is stable across runs. An unstable
        // order would change the prompt's cacheable prefix every round and
        // quietly cost more than the search gains.
        v.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.0.cmp(&b.id.0))
        });
        v.into_iter().take(budget).map(|l| l.text.clone()).collect()
    }

    /// Apply one mutation, subject to the frozen admission rules.
    ///
    /// The admission check happens here rather than at the call site because
    /// this is the only function that can change the state, and a check that
    /// can be bypassed by calling something else is not a check.
    pub fn apply(&mut self, m: &Mutation, admission: &Admission) -> Result<(), Refusal> {
        admission.admit(m)?;

        match m {
            Mutation::LessonAdd { text } => {
                let text = text.trim();
                // A blank lesson costs prompt tokens and teaches nothing;
                // a duplicate costs tokens twice.
                if !text.is_empty() && !self.lessons.iter().any(|l| l.text == text) {
                    self.lessons.push(Lesson {
                        id: LessonId(self.next_id),
                        text: text.to_string(),
                        weight: 1.0,
                    });
                    self.next_id += 1;
                }
            }
            Mutation::LessonRemove { id } => {
                self.lessons.retain(|l| l.id != *id);
            }
            Mutation::LessonReweight { id, delta } => {
                if let Some(l) = self.lessons.iter_mut().find(|l| l.id == *id) {
                    // Clamped rather than unbounded: a weight race would let
                    // one lesson crowd out every other regardless of budget.
                    l.weight = (l.weight + delta).clamp(0.0, 10.0);
                }
            }
            Mutation::ThresholdSet { knob, value } => {
                self.knobs.insert(*knob, *value);
            }
            // Level 2: the search rewriting its own move generator. Admission
            // has already checked the level is enabled; the invariants that
            // stop it promoting itself or emptying its vocabulary are in
            // `Grammar::apply_op`.
            Mutation::GrammarEdit { spec } => {
                let op: GrammarOp =
                    serde_json::from_value(spec.clone()).map_err(|e| Refusal::Unreadable {
                        detail: format!("not a grammar operation: {e}"),
                    })?;
                self.grammar
                    .apply_op(op)
                    .map_err(|detail| Refusal::Unreadable { detail })?;
            }
            // Level 3 is not a change to *this* state — it patches the search's
            // own source, which cannot be done by mutating a struct in memory.
            // It runs through `codepatch`, which builds and tests the patch in a
            // scratch tree and then asks a human. Reaching here means a caller
            // routed it to the wrong place.
            Mutation::CodePatch { .. } => {
                return Err(Refusal::Unreadable {
                    detail: "a code patch is not applied to policy state; \
                             route it through the level-3 gate"
                        .into(),
                });
            }
        }
        Ok(())
    }

    /// Apply a sequence, stopping at the first refusal.
    ///
    /// Returns how many took effect, so a caller can tell a rollout that did
    /// nothing from one that did everything.
    pub fn apply_all(
        &mut self,
        ms: &[Mutation],
        admission: &Admission,
    ) -> Result<usize, (usize, Refusal)> {
        for (i, m) in ms.iter().enumerate() {
            if let Err(e) = self.apply(m, admission) {
                return Err((i, e));
            }
        }
        Ok(ms.len())
    }
}

#[cfg(test)]
mod grammar_tests {
    use super::*;

    /// Level 2 enabled. Level 3 stays off unless a test says otherwise.
    fn admission() -> Admission {
        Admission::new(2)
    }

    fn edit(op: GrammarOp) -> Mutation {
        Mutation::GrammarEdit {
            spec: serde_json::to_value(op).unwrap(),
        }
    }

    #[test]
    fn a_grammar_edit_retunes_the_move_generator() {
        let mut s = PolicyState::default();
        assert_eq!(s.grammar().reweight_delta(), 0.5);
        s.apply(&edit(GrammarOp::SetReweightDelta { value: 1.5 }), &admission())
            .expect("a legal retune");
        assert_eq!(s.grammar().reweight_delta(), 1.5);
    }

    #[test]
    fn a_grammar_edit_can_narrow_the_vocabulary() {
        let mut s = PolicyState::default();
        assert!(s.grammar().kind_enabled("lesson_remove"));
        s.apply(
            &edit(GrammarOp::DisableKind { kind: "lesson_remove".into() }),
            &admission(),
        )
        .expect("narrowing is legal");
        assert!(!s.grammar().kind_enabled("lesson_remove"));
    }

    #[test]
    fn the_grammar_cannot_promote_itself_to_level_three() {
        // THE level-2 safety property. A machine that rewrites its own move
        // generator must not be able to write `code_patch` into its vocabulary
        // and thereby grant itself level 3.
        let mut s = PolicyState::default();
        let err = s
            .apply(
                &edit(GrammarOp::EnableKind { kind: "code_patch".into() }),
                &admission(),
            )
            .expect_err("enabling a level-3 move through the grammar must be refused");
        match err {
            Refusal::Unreadable { detail } => assert!(detail.contains("promote itself"), "{detail}"),
            other => panic!("unexpected refusal {other:?}"),
        }
        assert!(!s.grammar().kind_enabled("code_patch"));
    }

    #[test]
    fn the_grammar_cannot_be_emptied_into_a_no_op() {
        // A search with no moves scores as "changed nothing", which reads as
        // "no regression" — a degenerate optimum the search must not be able to
        // reach by disabling its own vocabulary.
        let mut s = PolicyState::default();
        let kinds: Vec<String> = s.grammar().enabled_kinds().map(|k| k.to_string()).collect();
        // Disable all but the last, which must be refused.
        for k in kinds.iter().take(kinds.len() - 1) {
            s.apply(&edit(GrammarOp::DisableKind { kind: k.clone() }), &admission())
                .expect("narrowing down to one is fine");
        }
        let last = kinds.last().unwrap().clone();
        assert!(
            s.apply(&edit(GrammarOp::DisableKind { kind: last }), &admission()).is_err(),
            "emptying the grammar must be refused"
        );
        assert_eq!(s.grammar().enabled_kinds().count(), 1);
    }

    #[test]
    fn nonsense_parameters_are_refused() {
        let mut s = PolicyState::default();
        for op in [
            GrammarOp::SetReweightDelta { value: f64::NAN },
            GrammarOp::SetReweightDelta { value: 0.0 },
            GrammarOp::SetDepth { value: 0 },
            GrammarOp::SetDepth { value: 1000 },
            GrammarOp::SetAdaptAlpha { value: -1.0 },
        ] {
            assert!(s.apply(&edit(op), &admission()).is_err());
        }
        // And none of them moved the state.
        assert_eq!(s.grammar(), &Grammar::default());
    }

    #[test]
    fn a_malformed_spec_is_refused_rather_than_ignored() {
        let mut s = PolicyState::default();
        let m = Mutation::GrammarEdit {
            spec: serde_json::json!({ "op": "not_a_real_operation" }),
        };
        assert!(s.apply(&m, &admission()).is_err());
    }

    #[test]
    fn a_grammar_edit_is_still_refused_when_level_two_is_off() {
        // The frozen admission gate is upstream of every invariant here: with
        // level 2 disabled the edit never reaches the grammar at all.
        let mut s = PolicyState::default();
        let off = Admission::new(1);
        assert!(s.apply(&edit(GrammarOp::SetDepth { value: 8 }), &off).is_err());
        assert_eq!(s.grammar().depth(), Grammar::default().depth());
    }

    #[test]
    fn a_code_patch_is_not_applied_to_policy_state() {
        let mut s = PolicyState::default();
        let m = Mutation::CodePatch { unified_diff: "--- a/x\n+++ b/x\n".into() };
        // Even with level 3 admitted, this state is the wrong place for it.
        assert!(s.apply(&m, &Admission::new(3)).is_err());
    }

    #[test]
    fn the_grammar_is_part_of_the_state_identity() {
        // A candidate that differs only in its grammar must not hash the same
        // as the incumbent, or the certificate would attribute one's outcomes
        // to the other.
        let a = PolicyState::default();
        let mut b = PolicyState::default();
        b.apply(&edit(GrammarOp::SetDepth { value: 9 }), &admission()).unwrap();
        assert_ne!(a.digest(), b.digest());
    }
}
