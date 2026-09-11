//! The frozen core.
//!
//! Everything in this crate is outside the machine's reach. There is no
//! mutation API here, nothing in [`samaritan_dsl::Mutation`] can name any of
//! these items, and `crates/samaritan-kernel/` is on [`FROZEN_PATHS`], so a
//! level-3 code patch that touched this crate would be refused before it was
//! ever scored.
//!
//! Three things live here and nothing else:
//!
//! - [`tier`] — how an action is routed. The leash.
//! - [`admission`] — what the machine may change about itself. The leash on
//!   changing the leash.
//! - [`utility`] — what counts as doing well, with safety made
//!   non-negotiable rather than merely expensive.
//! - [`budget`] — a fixed compute allowance the agent cannot raise, which
//!   turns wasted effort into a real cost rather than a reported statistic.
//!
//! The recurring principle: the component being optimised must not be able to
//! reach the component that defines what optimising means.

pub mod admission;
pub mod budget;
pub mod tier;
pub mod utility;

pub use admission::{Admission, FROZEN_PATHS, Refusal, is_breach, knob_range};
pub use budget::{ComputeBudget, ComputeTier};
pub use tier::{ActionClass, AutonomyCeiling, Promotions, Tier, base_tier, route};
pub use utility::{Components, EpisodeUtility, Violation, Weights};

/// Approval, asked of a human.
///
/// A trait so the TUI, a test double, and a non-interactive runner can each
/// supply one — but note what it cannot express: there is no variant that
/// grants standing permission, and no way for an implementation to answer on
/// the human's behalf. A [`Tier::Confirm`] action blocks until this returns.
pub trait ApprovalGate {
    /// Ask about one action. `Err` means the question could not be put to
    /// anyone, which callers must treat as a refusal and never as an approval.
    fn ask(&mut self, request: &ApprovalRequest) -> Result<Approval, GateError>;
}

/// What the human is shown before answering.
#[derive(Debug, Clone, PartialEq)]
pub struct ApprovalRequest<'a> {
    pub action: &'a samaritan_dsl::ProposedAction,
    /// Why the agent wants this, taken from the decision record.
    pub rationale: &'a str,
    /// What the agent expects to happen, so the human can disagree with the
    /// prediction rather than only with the action.
    pub prediction: &'a samaritan_dsl::Prediction,
    pub tier: Tier,
}

/// The answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    Allow,
    Refuse,
}

#[derive(Debug, thiserror::Error)]
pub enum GateError {
    #[error("no human available to ask")]
    Unattended,
    #[error("the approval channel failed: {0}")]
    Channel(String),
}

/// Compile-time guard: the gate must be usable across threads, and `Approval`
/// must stay a plain two-state answer. If someone adds an `AllowAlways`
/// variant, this stops compiling and the review conversation happens.
const _: () = {
    assert!(std::mem::size_of::<Approval>() == 1);
};
