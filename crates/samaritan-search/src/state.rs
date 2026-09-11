//! The mutable half of the machine: what a self-modification may change.
//!
//! Everything here is searchable. Everything the search must not reach lives
//! in `samaritan-kernel`, which exposes no mutation API, so the partition is
//! enforced by the type system rather than by remembering to check.

use std::collections::BTreeMap;

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

/// The searchable state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyState {
    lessons: Vec<Lesson>,
    knobs: BTreeMap<Knob, f64>,
    next_id: u64,
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
        }
    }
}

impl PolicyState {
    pub fn lessons(&self) -> &[Lesson] {
        &self.lessons
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
            // Levels 2 and 3. Admission refuses these while they are off, so
            // reaching here means a human enabled them and the applier has
            // not caught up yet — which must fail loudly rather than silently
            // doing nothing.
            Mutation::GrammarEdit { .. } | Mutation::CodePatch { .. } => {
                return Err(Refusal::Unreadable {
                    detail: "level 2 and 3 mutations are admitted but not yet applicable".into(),
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
