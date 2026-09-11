//! Compute as a binding constraint rather than a statistic.
//!
//! `diagnose` can *detect* the degenerate attractor — both agents optimising
//! the game while capability rots — but detection after the fact is a poor
//! substitute for a pressure that prevents it. Conway's Automaton makes this
//! point well by construction: its agents buy their own inference, so one that
//! stops producing value runs out of compute and stops. The selection is not a
//! rule anyone enforces; it is arithmetic.
//!
//! The same arithmetic applies here without any of the economics. Every
//! episode and every self-modification attempt spends from a fixed allowance.
//! An agent farming exploits or re-treading a known trick is spending real
//! budget for no movement on the yardstick, and the budget does not come back.
//! As it depletes, self-modification is switched off before task work is,
//! which forces the remaining compute toward the thing being measured.
//!
//! The allowance is frozen. There is deliberately no method on
//! [`ComputeBudget`] that increases it: an optimiser that can print its own
//! compute is not under a constraint, it is under a suggestion.

use serde::{Deserialize, Serialize};

/// How much room the agent has left, in coarse bands.
///
/// Bands rather than a raw fraction because the behaviour changes at
/// thresholds, and a threshold that is computed in three places will
/// eventually be computed differently in one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeTier {
    /// Over half the allowance remains. Everything is permitted.
    Full,
    /// Under half. Cheaper inference, fewer lessons carried into each prompt.
    Reduced,
    /// Under a fifth. Task work only — self-modification is switched off,
    /// because a speculative rewrite is the least defensible way to spend the
    /// last of a budget that is about to run out.
    Critical,
    /// Nothing left. No further work of any kind.
    Exhausted,
}

impl ComputeTier {
    /// Whether the search may attempt self-modification at this tier.
    pub fn allows_self_modification(self) -> bool {
        matches!(self, ComputeTier::Full | ComputeTier::Reduced)
    }

    /// Whether any episode may run at all.
    pub fn allows_work(self) -> bool {
        self != ComputeTier::Exhausted
    }

    /// How many lessons to carry into a decision prompt, as a multiplier on
    /// the configured budget. Cheaper prompts buy more episodes from what is
    /// left.
    pub fn lesson_budget_scale(self) -> f64 {
        match self {
            ComputeTier::Full => 1.0,
            ComputeTier::Reduced => 0.5,
            ComputeTier::Critical => 0.25,
            ComputeTier::Exhausted => 0.0,
        }
    }
}

/// A fixed allowance of inference, and what has been spent against it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeBudget {
    /// Set once, by a human, at the start of a run. Private and with no
    /// setter: this is the constraint, and a constraint the optimiser can
    /// reach is not one.
    allotted_tokens: u64,
    spent_tokens: u64,
}

impl ComputeBudget {
    pub fn new(allotted_tokens: u64) -> Self {
        Self {
            allotted_tokens,
            spent_tokens: 0,
        }
    }

    /// Record spending. Saturating, so an over-spend clamps to exhausted
    /// rather than wrapping into an enormous remaining balance.
    pub fn spend(&mut self, tokens: u64) {
        self.spent_tokens = self.spent_tokens.saturating_add(tokens);
    }

    pub fn allotted(&self) -> u64 {
        self.allotted_tokens
    }

    pub fn spent(&self) -> u64 {
        self.spent_tokens
    }

    pub fn remaining(&self) -> u64 {
        self.allotted_tokens.saturating_sub(self.spent_tokens)
    }

    pub fn fraction_remaining(&self) -> f64 {
        if self.allotted_tokens == 0 {
            return 0.0;
        }
        self.remaining() as f64 / self.allotted_tokens as f64
    }

    pub fn tier(&self) -> ComputeTier {
        if self.remaining() == 0 {
            return ComputeTier::Exhausted;
        }
        match self.fraction_remaining() {
            f if f > 0.5 => ComputeTier::Full,
            f if f > 0.2 => ComputeTier::Reduced,
            _ => ComputeTier::Critical,
        }
    }

    /// Whether a proposed spend fits in what is left.
    ///
    /// Checked before an episode rather than discovered during one, so a run
    /// ends at a round boundary with a complete ledger instead of halfway
    /// through an unrecorded episode.
    pub fn affords(&self, tokens: u64) -> bool {
        self.remaining() >= tokens
    }
}
