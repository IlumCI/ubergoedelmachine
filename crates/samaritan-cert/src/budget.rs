//! Spending the error budget across a lifetime of tests.
//!
//! One martingale controls the error of one decision. A harness that commits
//! thousands of self-modifications needs the error controlled across all of
//! them, and the obvious construction is the wrong one.
//!
//! **Geometric spending** — give test *k* a budget of `δ·2⁻ᵏ`, so the sum
//! stays under `δ` — is a correct union bound and a terrible instrument. By
//! the thirtieth commit the budget is `δ/10⁹`, which no realistic amount of
//! evidence can clear. A run that improves steadily would simply stop being
//! able to prove it. It is kept here as [`Spending::Geometric`] because it is
//! the right tool for a handful of irreversible decisions — level-3 code
//! rewrites — where the count is small and a bad commit is expensive.
//!
//! **α-investing** (Foster & Stine) is the instrument for the common case.
//! The budget is wealth rather than a schedule: each test spends from it, and
//! a *rejection pays out*. A run that keeps finding real improvements keeps
//! funding its own testing; a run that keeps proposing rubbish runs out and
//! stops. That is the right incentive, and it controls the marginal false
//! discovery rate rather than the family-wise error rate — which is the
//! honest thing to control when the intent is to accept many discoveries
//! rather than to avoid any false one.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Spending {
    /// α-investing. Default for levels 1 and 2.
    Investing {
        /// Starting wealth, and the cap on a single payout.
        alpha0: f64,
        /// Fraction of current wealth staked on each test.
        stake: f64,
    },
    /// Summable schedule, `δ·2⁻ᵏ`. Strict FWER, starves quickly.
    Geometric { delta: f64 },
}

impl Default for Spending {
    fn default() -> Self {
        Spending::Investing {
            alpha0: 0.05,
            stake: 0.5,
        }
    }
}

/// The error budget for one level of the search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    policy: Spending,
    wealth: f64,
    tests: u64,
    commits: u64,
    spent_total: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BudgetError {
    #[error("the error budget is exhausted; no further modification can be certified")]
    Exhausted,
    #[error("alpha must be in (0, 1); got {0}")]
    BadAlpha(f64),
}

impl Budget {
    pub fn new(policy: Spending) -> Result<Self, BudgetError> {
        let wealth = match policy {
            Spending::Investing { alpha0, .. } => alpha0,
            Spending::Geometric { delta } => delta,
        };
        if !(0.0..1.0).contains(&wealth) || wealth <= 0.0 {
            return Err(BudgetError::BadAlpha(wealth));
        }
        Ok(Self {
            policy,
            wealth,
            tests: 0,
            commits: 0,
            spent_total: 0.0,
        })
    }

    pub fn wealth(&self) -> f64 {
        self.wealth
    }
    pub fn tests(&self) -> u64 {
        self.tests
    }
    pub fn commits(&self) -> u64 {
        self.commits
    }
    /// Total α committed across the run. Under geometric spending this is
    /// bounded by δ by construction; under investing it is not, and is not
    /// meant to be — the guarantee there is on the rate, not the sum.
    pub fn spent_total(&self) -> f64 {
        self.spent_total
    }
}

/// Below this fraction of the starting wealth, a budget is spent.
///
/// Proportional staking never reaches zero — it halves, and halves again —
/// so without a floor the budget would report itself usable forever while
/// quietly demanding an e-value of twenty thousand. An alpha nothing can
/// clear is not a smaller alpha, it is a stopped process, and saying so is
/// more useful than letting a search grind against a threshold it cannot
/// reach.
const EXHAUSTED_BELOW: f64 = 1e-3;

impl Budget {
    /// The significance level for the next test.
    ///
    /// Deliberately not a free parameter at the call site: a caller that
    /// could choose its own α would choose a generous one on the test it
    /// cared about.
    pub fn next_alpha(&self) -> Result<f64, BudgetError> {
        let a = match self.policy {
            Spending::Investing { stake, .. } => self.wealth * stake.clamp(0.01, 0.99),
            Spending::Geometric { delta } => delta / 2f64.powi(self.tests as i32 + 1),
        };
        if a <= 0.0 || !a.is_finite() || self.below_floor() {
            return Err(BudgetError::Exhausted);
        }
        Ok(a)
    }

    /// Record the outcome of a test that spent `alpha`.
    ///
    /// A rejection pays back into the wealth, capped at `alpha0`: a single
    /// lucky discovery must not fund an unbounded run of subsequent tests.
    pub fn record(&mut self, alpha: f64, committed: bool) {
        self.tests += 1;
        self.spent_total += alpha;

        match self.policy {
            Spending::Investing { alpha0, .. } => {
                self.wealth -= alpha;
                if committed {
                    self.commits += 1;
                    self.wealth = (self.wealth + alpha0).min(alpha0 * 2.0);
                }
                self.wealth = self.wealth.max(0.0);
            }
            Spending::Geometric { .. } => {
                // The schedule does not depend on outcomes; wealth is
                // tracked only so `exhausted` means something.
                self.wealth = (self.wealth - alpha).max(0.0);
                if committed {
                    self.commits += 1;
                }
            }
        }
    }

    /// Whether the remaining wealth is too small to fund a real test.
    fn below_floor(&self) -> bool {
        let start = match self.policy {
            Spending::Investing { alpha0, .. } => alpha0,
            Spending::Geometric { delta } => delta,
        };
        self.wealth < start * EXHAUSTED_BELOW
    }

    /// Whether any further test can be funded.
    pub fn exhausted(&self) -> bool {
        self.below_floor() || self.next_alpha().is_err()
    }
}
