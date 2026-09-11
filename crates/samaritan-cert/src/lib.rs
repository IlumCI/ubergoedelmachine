//! Deciding whether a self-modification may be kept.
//!
//! Schmidhuber's Gödel machine rewrites itself only on a *proof* that the
//! rewrite raises expected utility. Nobody has built one, because the proof
//! search is the intractable part. This crate is the substitution that makes
//! the idea buildable: a sequential statistical certificate in place of a
//! proof, sound with probability `1 − α` instead of with certainty, and
//! bought with sampling instead of deduction.
//!
//! The three pieces:
//!
//! - [`martingale`] — evidence that a candidate beats the incumbent, valid at
//!   any stopping time.
//! - [`budget`] — how error is spent across a lifetime of such decisions.
//! - [`Certificate`] — the decision, plus the checks that stop the evidence
//!   from being meaningless in the first place.
//!
//! That last part matters more than it sounds. A martingale computed on the
//! wrong tasks is arithmetic, not evidence, and it produces a number that
//! looks exactly like the real thing.

pub mod budget;
pub mod martingale;

use serde::{Deserialize, Serialize};

pub use budget::{Budget, BudgetError, Spending};
pub use martingale::{Betting, Martingale, Pair};

/// Which tasks a paired sample came from.
///
/// Carried through to the decision so the certificate can refuse a sample
/// drawn from tasks the proposer already saw. The candidate was *proposed* by
/// a search that had looked at performance, so testing it on the tasks that
/// inspired it is circular — the evidence would be real arithmetic about a
/// question nobody asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Tasks the search has optimised against. Never valid for a certificate.
    Train,
    /// Held out, and strictly later in time than anything the search saw.
    HeldOut,
}

/// Why a modification was not certified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Refusal {
    /// The evidence did not reach the threshold. The ordinary outcome, and
    /// not a failure of anything.
    Insufficient { e_value: f64, needed: f64 },
    /// The sample came from tasks the proposer had already seen.
    Circular,
    /// Too few informative pairs to conclude anything.
    ///
    /// Distinct from `Insufficient`: that means the evidence went the wrong
    /// way, this means there was effectively none. A candidate that solves
    /// exactly the same tasks as the incumbent produces no discordant pairs
    /// at all, however many tasks are run.
    TooFewDiscordant { discordant: u64, needed: u64 },
    /// The error budget is spent.
    BudgetExhausted,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Insufficient { e_value, needed } => {
                write!(f, "evidence {e_value:.2} did not reach {needed:.2}")
            }
            Refusal::Circular => write!(
                f,
                "the sample came from tasks the proposer optimised against"
            ),
            Refusal::TooFewDiscordant { discordant, needed } => write!(
                f,
                "only {discordant} informative pairs; {needed} required"
            ),
            Refusal::BudgetExhausted => write!(f, "the error budget is exhausted"),
        }
    }
}

/// A granted certificate. Holding one is the authority to commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Granted {
    /// Evidence at the moment of crossing.
    pub e_value: f64,
    /// The level this test was run at.
    pub alpha: f64,
    pub discordant: u64,
    pub concordant: u64,
    pub candidate_wins: u64,
    pub pairs: u64,
}

/// Minimum informative pairs before any conclusion is drawn.
///
/// A floor on top of the martingale rather than a substitute for it. With a
/// large enough λ the wealth can cross on three lucky pairs; that crossing is
/// *valid* — the type-I error is still bounded — but it is a decision nobody
/// would want to act on, and the search would spend its budget on noise.
const MIN_DISCORDANT: u64 = 12;

/// The gate a self-modification has to pass.
pub struct Certificate {
    budget: Budget,
    betting: Betting,
    min_discordant: u64,
}

impl Certificate {
    pub fn new(spending: Spending) -> Result<Self, BudgetError> {
        Ok(Self {
            budget: Budget::new(spending)?,
            betting: Betting::Adaptive { cap: 0.5 },
            min_discordant: MIN_DISCORDANT,
        })
    }

    pub fn with_betting(mut self, betting: Betting) -> Self {
        self.betting = betting;
        self
    }

    pub fn with_min_discordant(mut self, n: u64) -> Self {
        self.min_discordant = n;
        self
    }

    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// Test one candidate against the incumbent.
    ///
    /// Consumes budget whether or not the modification is certified — a test
    /// that fails still spent the look, and pretending otherwise is how a
    /// search talks itself into an unbounded number of attempts.
    pub fn test(
        &mut self,
        pairs: &[Pair],
        provenance: Provenance,
    ) -> Result<Granted, Refusal> {
        // Checked before anything is spent: a circular sample is not a failed
        // test, it is a test that was never valid, and charging for it would
        // punish the caller for a mistake the caller should be told about.
        if provenance != Provenance::HeldOut {
            return Err(Refusal::Circular);
        }

        let alpha = self.budget.next_alpha().map_err(|_| Refusal::BudgetExhausted)?;

        let mut m = Martingale::new(self.betting);
        m.observe_all(pairs);

        if m.discordant() < self.min_discordant {
            // Not charged: there was no test here to pay for.
            return Err(Refusal::TooFewDiscordant {
                discordant: m.discordant(),
                needed: self.min_discordant,
            });
        }

        let crossed = m.crossed(alpha);
        self.budget.record(alpha, crossed);

        if crossed {
            Ok(Granted {
                e_value: m.e_value(),
                alpha,
                discordant: m.discordant(),
                concordant: m.concordant(),
                candidate_wins: m.candidate_wins(),
                pairs: pairs.len() as u64,
            })
        } else {
            Err(Refusal::Insufficient {
                e_value: m.e_value(),
                needed: 1.0 / alpha,
            })
        }
    }
}
