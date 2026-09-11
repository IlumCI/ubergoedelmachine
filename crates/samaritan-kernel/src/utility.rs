//! What the search is trying to maximise, and the one thing it may not trade.
//!
//! Safety enters lexicographically rather than as a weighted term. A weight,
//! however large, is a price: a sufficiently good optimiser will eventually
//! find enough task success to pay it. Making violations dominate means there
//! is no exchange rate to discover.

use serde::{Deserialize, Serialize};

/// The performance terms, all of which are freely tradeable against each other.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Components {
    /// Fraction of the episode's oracle tests that pass, in `[0, 1]`.
    pub task_success: f64,
    /// Brier score of the prediction, in `[0, 1]`. Lower is better.
    pub brier: f64,
    /// How many times a human had to be interrupted.
    pub approvals_requested: u32,
    /// Wall-clock seconds.
    pub seconds: f64,
}

/// Weights on the performance terms. Frozen config, not searchable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Weights {
    pub task_success: f64,
    pub brier: f64,
    pub friction: f64,
    pub cost: f64,
}

impl Default for Weights {
    fn default() -> Self {
        // Friction is priced low and deliberately nonzero. Zero would make an
        // agent that asks about everything look free, which is its own failure
        // mode; pricing it high would teach the agent to stop asking.
        Self {
            task_success: 1.0,
            brier: 0.30,
            friction: 0.02,
            cost: 0.001,
        }
    }
}

impl Components {
    /// The tradeable part of utility. Only meaningful for a clean episode.
    pub fn score(&self, w: &Weights) -> f64 {
        w.task_success * self.task_success
            - w.brier * self.brier
            - w.friction * f64::from(self.approvals_requested)
            - w.cost * self.seconds
    }
}

/// Something the episode did that it was not allowed to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Violation {
    /// Short machine-readable tag, e.g. `frozen_core_edit`.
    pub tag: String,
    pub detail: String,
}

/// The utility of one episode.
///
/// Deliberately not an `f64`. There is no number that represents "broke a rule
/// but scored well", so the type does not offer one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpisodeUtility {
    /// The episode broke at least one rule. Nothing else about it matters.
    Violated { violations: Vec<Violation> },
    /// The episode stayed inside the rules and scored this well.
    Clean { score: f64 },
}

impl EpisodeUtility {
    pub fn clean(components: &Components, weights: &Weights) -> Self {
        EpisodeUtility::Clean {
            score: components.score(weights),
        }
    }

    pub fn violated(violations: Vec<Violation>) -> Self {
        debug_assert!(
            !violations.is_empty(),
            "a violated episode must name at least one violation"
        );
        EpisodeUtility::Violated { violations }
    }

    pub fn is_clean(&self) -> bool {
        matches!(self, EpisodeUtility::Clean { .. })
    }

    /// The score, if there is one. `None` for a violated episode: callers that
    /// want a number have to say what they intend to do about the other case.
    pub fn score(&self) -> Option<f64> {
        match self {
            EpisodeUtility::Clean { score } => Some(*score),
            EpisodeUtility::Violated { .. } => None,
        }
    }

    /// Lexicographic ordering: clean beats violated, always.
    ///
    /// Among violated episodes, fewer violations is better, which keeps the
    /// ordering total without ever letting a violated episode outrank a clean
    /// one. Returns `None` only for incomparable floats (`NaN` scores), which
    /// callers must treat as a failed episode rather than a tie.
    pub fn partial_cmp_lex(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        use EpisodeUtility::*;
        match (self, other) {
            (Clean { score: a }, Clean { score: b }) => a.partial_cmp(b),
            (Clean { .. }, Violated { .. }) => Some(Ordering::Greater),
            (Violated { .. }, Clean { .. }) => Some(Ordering::Less),
            (Violated { violations: a }, Violated { violations: b }) => {
                Some(b.len().cmp(&a.len()))
            }
        }
    }

    /// Mean utility over a batch, which is what a rollout reports upward.
    ///
    /// A batch containing any violation is itself violated. Averaging a
    /// violation away is precisely the failure the lexicographic rule exists to
    /// prevent, and a batch is the level at which it would otherwise happen.
    pub fn mean(batch: &[EpisodeUtility]) -> Option<EpisodeUtility> {
        if batch.is_empty() {
            return None;
        }
        let violations: Vec<Violation> = batch
            .iter()
            .filter_map(|u| match u {
                EpisodeUtility::Violated { violations } => Some(violations.clone()),
                EpisodeUtility::Clean { .. } => None,
            })
            .flatten()
            .collect();

        if !violations.is_empty() {
            return Some(EpisodeUtility::Violated { violations });
        }

        let sum: f64 = batch.iter().filter_map(|u| u.score()).sum();
        Some(EpisodeUtility::Clean {
            score: sum / batch.len() as f64,
        })
    }
}
