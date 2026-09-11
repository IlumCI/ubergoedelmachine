//! A test martingale for "is the candidate actually better".
//!
//! This is the component that stands in for Schmidhuber's proof. A Gödel
//! machine commits a self-modification only on a demonstration that it raises
//! expected utility; proof search is intractable, so the demonstration here is
//! statistical — and the whole design rests on that substitution being sound
//! rather than merely plausible.
//!
//! # The construction
//!
//! Candidate and incumbent run **paired** on the same tasks, which blocks out
//! task difficulty: whether a problem was hard is common to both arms and
//! cancels. Each task yields two bits — did the candidate solve it, did the
//! incumbent — and the pairs split three ways:
//!
//! - both solved, or neither: **concordant**, and carries no information
//!   about a *difference*. Discarded, exactly as McNemar's test discards it.
//! - candidate solved and incumbent did not: evidence **for**.
//! - incumbent solved and candidate did not: evidence **against**.
//!
//! Under the null "the candidate is no better", the probability that a
//! discordant pair favours the candidate is at most one half. So bet on it.
//! Starting from wealth 1 and betting a fraction λ of the pot on each
//! discordant pair:
//!
//! ```text
//!     W ← W · (1 + λ)   when the candidate wins the pair
//!     W ← W · (1 − λ)   when the incumbent does
//! ```
//!
//! Under the null the expected multiplier is `p(1+λ) + (1−p)(1−λ)`
//! `= 1 + λ(2p − 1) ≤ 1` for `p ≤ ½`, so `W` is a non-negative
//! supermartingale starting at 1. **Ville's inequality** then gives
//! `P(∃t : W_t ≥ 1/α) ≤ α`.
//!
//! That last line is the entire point, and it is worth being precise about
//! what it buys: the guarantee is over the *whole run*, at *any* stopping
//! time. You may watch the wealth climb, stop the moment it crosses, and add
//! more pairs when it does not — none of which is permitted with a p-value.
//! A search that peeks at its own evidence is exactly what this harness does,
//! so anytime validity is not a refinement here, it is the requirement.
//!
//! # What it does not give you
//!
//! Validity is over stopping times, not over hypotheses chosen after seeing
//! the data. The candidate was *proposed* by a search that had already looked
//! at performance, so testing it on the tasks that inspired it would be
//! circular. The pairs fed in here must come from tasks the proposer never
//! saw; [`crate::Certificate`] refuses to accept them otherwise.

use serde::{Deserialize, Serialize};

/// One paired observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pair {
    pub candidate_solved: bool,
    pub incumbent_solved: bool,
}

impl Pair {
    pub fn new(candidate_solved: bool, incumbent_solved: bool) -> Self {
        Self {
            candidate_solved,
            incumbent_solved,
        }
    }

    /// `None` when concordant, and therefore uninformative about a difference.
    pub fn favours_candidate(&self) -> Option<bool> {
        match (self.candidate_solved, self.incumbent_solved) {
            (true, false) => Some(true),
            (false, true) => Some(false),
            _ => None,
        }
    }
}

/// How much of the pot to stake on each pair.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Betting {
    /// A constant fraction. Simple, and a reasonable default at 0.5.
    Fixed(f64),
    /// Stake in proportion to the edge observed *so far*.
    ///
    /// The bet must be **predictable** — chosen from the pairs already seen,
    /// never including the one being bet on — or the supermartingale property
    /// is lost and with it the guarantee. Hence `seen`/`wins` are the counts
    /// *before* the current pair.
    ///
    /// Capped below 1: a stake of 1 would bankrupt the process on a single
    /// adverse pair, and a wealth of exactly zero can never recover.
    Adaptive { cap: f64 },
}

impl Betting {
    fn lambda(&self, wins: u64, seen: u64) -> f64 {
        match self {
            Betting::Fixed(l) => l.clamp(0.0, 0.95),
            Betting::Adaptive { cap } => {
                if seen == 0 {
                    return 0.25;
                }
                let p = wins as f64 / seen as f64;
                // 2p − 1 is the empirical edge over the null's ½. Negative
                // edge means bet nothing rather than bet against ourselves:
                // this is a one-sided test of "better", not of "different".
                (2.0 * p - 1.0).clamp(0.0, cap.clamp(0.0, 0.95))
            }
        }
    }
}

/// Accumulated evidence that the candidate beats the incumbent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Martingale {
    wealth: f64,
    /// Highest wealth ever reached. Ville's inequality is about *ever*
    /// crossing, so a threshold crossed and then fallen back through still
    /// counts — which is why the running maximum is tracked rather than
    /// recomputed from the current value.
    peak: f64,
    discordant: u64,
    candidate_wins: u64,
    concordant: u64,
    betting: Betting,
}

impl Martingale {
    pub fn new(betting: Betting) -> Self {
        Self {
            wealth: 1.0,
            peak: 1.0,
            discordant: 0,
            candidate_wins: 0,
            concordant: 0,
            betting,
        }
    }

    /// Feed one paired result.
    pub fn observe(&mut self, pair: Pair) {
        let Some(candidate_won) = pair.favours_candidate() else {
            self.concordant += 1;
            return;
        };

        // Predictable: computed from what came before this pair.
        let lambda = self.betting.lambda(self.candidate_wins, self.discordant);

        self.wealth *= if candidate_won {
            1.0 + lambda
        } else {
            1.0 - lambda
        };
        self.peak = self.peak.max(self.wealth);

        self.discordant += 1;
        if candidate_won {
            self.candidate_wins += 1;
        }
    }

    pub fn observe_all(&mut self, pairs: &[Pair]) {
        for p in pairs {
            self.observe(*p);
        }
    }

    pub fn wealth(&self) -> f64 {
        self.wealth
    }

    /// The most evidence ever accumulated, which is what the guarantee is
    /// about.
    pub fn peak(&self) -> f64 {
        self.peak
    }

    pub fn discordant(&self) -> u64 {
        self.discordant
    }

    pub fn concordant(&self) -> u64 {
        self.concordant
    }

    pub fn candidate_wins(&self) -> u64 {
        self.candidate_wins
    }

    /// Whether the evidence has *ever* crossed `1/alpha`.
    ///
    /// Uses the peak rather than the current wealth on purpose: Ville bounds
    /// the probability of ever crossing, so a run that crossed and then
    /// drifted back down has still spent its evidence and is entitled to
    /// stop.
    pub fn crossed(&self, alpha: f64) -> bool {
        alpha > 0.0 && self.peak >= 1.0 / alpha
    }

    /// The e-value: evidence against the null, on a scale where `1` is none.
    pub fn e_value(&self) -> f64 {
        self.peak
    }
}
