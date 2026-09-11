//! The execution router: the thing that stands between a decision and the
//! world.
//!
//! `samaritan-kernel` decides *what tier* an action belongs to; this crate
//! acts on that decision — asks the human when asking is required, executes
//! when it is not, refuses when it must, and writes all of it to the ledger.
//! It is where the four other crates meet, and it is deliberately the only
//! place in the system where a `ProposedAction` turns into something that
//! happens.
//!
//! Three rules run through it, each of which exists because the obvious
//! alternative fails quietly rather than loudly:
//!
//! 1. **A gate failure is a refusal.** If the human cannot be reached, the
//!    action does not run. Treating "nobody answered" as consent would make
//!    the whole gate depend on the approval channel staying up, which is
//!    exactly the condition an adversary would arrange.
//! 2. **A misreported action is a violation.** Everything downstream trusts
//!    the labels an agent puts on its own actions. An agent that understates
//!    one has not made a mistake about the world, it has made the routing
//!    decision meaningless, so it is charged lexicographically rather than
//!    merely scored down.
//! 3. **The leash shortens without ceremony.** A violation revokes earned
//!    autonomy immediately and resets the streak. Only lengthening requires
//!    evidence.

use std::collections::BTreeMap;

use samaritan_dsl::{Authority, DecisionRecord, ProposedAction};
use samaritan_exec::{Executor, Outcome};
use samaritan_kernel::{
    ActionClass, Approval, ApprovalGate, ApprovalRequest, AutonomyCeiling, ComputeBudget,
    ComputeTier, Promotions, Tier, Violation, route,
};
use samaritan_ledger::{Actor, Event, Ledger, LedgerError};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("ledger error: {0}")]
    Ledger(#[from] LedgerError),

    #[error("compute exhausted; no further work may run")]
    ComputeExhausted,
}

/// What happened to one action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionDisposition {
    /// Ran without asking.
    Executed { succeeded: bool },
    /// A human was asked and said yes; then it ran.
    Approved { succeeded: bool },
    /// A human was asked and said no.
    Refused,
    /// The human could not be reached. Treated as a refusal.
    Unattended { detail: String },
    /// The frozen rules said no. Nobody was asked, because it was not a
    /// question.
    Denied,
}

impl ActionDisposition {
    pub fn ran(&self) -> bool {
        matches!(
            self,
            ActionDisposition::Executed { .. } | ActionDisposition::Approved { .. }
        )
    }
}

/// The result of routing one action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionResult {
    pub index: usize,
    pub tier: Tier,
    pub disposition: ActionDisposition,
    /// Absent when the action never ran.
    pub outcome: Option<Outcome>,
}

/// The result of routing every action in one decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dispatch {
    pub results: Vec<ActionResult>,
    /// Everything that went wrong in a way the utility function must treat as
    /// dominant: sandbox escapes, malformed payloads, misreported actions.
    pub violations: Vec<Violation>,
    /// How many times a human was interrupted. Feeds the friction term.
    pub approvals_requested: u32,
}

impl Dispatch {
    pub fn clean(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Routes a decision's actions, and remembers what it has learned about
/// whether each class of action deserves less supervision.
pub struct Router {
    ceiling: AutonomyCeiling,
    promotions: Promotions,
    budget: ComputeBudget,
    /// Consecutive approvals per class with no violation in between.
    streaks: BTreeMap<ActionClass, u32>,
}

impl Router {
    pub fn new(ceiling: AutonomyCeiling, budget: ComputeBudget) -> Self {
        Self {
            ceiling,
            promotions: Promotions::none(),
            budget,
            streaks: BTreeMap::new(),
        }
    }

    pub fn budget(&self) -> &ComputeBudget {
        &self.budget
    }

    pub fn promotions(&self) -> &Promotions {
        &self.promotions
    }

    pub fn streak(&self, class: ActionClass) -> u32 {
        self.streaks.get(&class).copied().unwrap_or(0)
    }

    /// Charge inference against the budget.
    pub fn spend(&mut self, tokens: u64) {
        self.budget.spend(tokens);
    }

    /// Route and carry out every action in a decision.
    ///
    /// The record's [`Authority`] applies to all of its actions: a decision is
    /// tainted as a whole, because an agent persuaded by adversary-written
    /// content cannot be trusted to have kept that influence to one action.
    pub fn dispatch(
        &mut self,
        record: &DecisionRecord,
        executor: &mut Executor,
        gate: &mut dyn ApprovalGate,
        ledger: &mut Ledger,
    ) -> Result<Dispatch, RouterError> {
        if !self.budget.tier().allows_work() {
            return Err(RouterError::ComputeExhausted);
        }

        ledger.append(
            Actor::Warden,
            &Event::DecisionRecorded {
                record: Box::new(record.clone()),
            },
        )?;

        let mut results = Vec::new();
        let mut violations = Vec::new();
        let mut approvals_requested = 0u32;

        for (index, action) in record.actions().iter().enumerate() {
            let tier = route(action, record.authority(), &self.ceiling, &self.promotions);
            ledger.append(
                Actor::System,
                &Event::ActionRouted {
                    decision: record.id(),
                    action_index: index,
                    tier,
                },
            )?;

            let (disposition, outcome) = match tier {
                Tier::Deny => (ActionDisposition::Denied, None),

                Tier::Auto => {
                    let out = executor.execute(action);
                    (
                        ActionDisposition::Executed {
                            succeeded: out.succeeded,
                        },
                        Some(out),
                    )
                }

                Tier::Confirm => {
                    approvals_requested += 1;
                    let request = ApprovalRequest {
                        action,
                        rationale: record.rationale(),
                        prediction: record.prediction(),
                        tier,
                    };
                    let started = std::time::Instant::now();
                    let answer = gate.ask(&request);
                    let waited = started.elapsed().as_secs_f64();

                    match answer {
                        Ok(Approval::Allow) => {
                            ledger.append(
                                Actor::Human,
                                &Event::ApprovalResolved {
                                    decision: record.id(),
                                    action_index: index,
                                    allowed: true,
                                    waited_secs: waited,
                                },
                            )?;
                            let out = executor.execute(action);
                            (
                                ActionDisposition::Approved {
                                    succeeded: out.succeeded,
                                },
                                Some(out),
                            )
                        }
                        Ok(Approval::Refuse) => {
                            ledger.append(
                                Actor::Human,
                                &Event::ApprovalResolved {
                                    decision: record.id(),
                                    action_index: index,
                                    allowed: false,
                                    waited_secs: waited,
                                },
                            )?;
                            (ActionDisposition::Refused, None)
                        }
                        // The rule that makes the gate worth having. An
                        // unreachable human is not a permissive one; if this
                        // branch ever executed the action, taking down the
                        // approval channel would become the cheapest possible
                        // exploit.
                        Err(e) => {
                            ledger.append(
                                Actor::System,
                                &Event::ApprovalResolved {
                                    decision: record.id(),
                                    action_index: index,
                                    allowed: false,
                                    waited_secs: waited,
                                },
                            )?;
                            (
                                ActionDisposition::Unattended {
                                    detail: e.to_string(),
                                },
                                None,
                            )
                        }
                    }
                }
            };

            if let Some(out) = &outcome {
                violations.extend(out.violations.iter().cloned());

                if out.understated(action) {
                    violations.push(misgrade_violation(action, out));
                }

                ledger.append(
                    Actor::Warden,
                    &Event::ActionExecuted {
                        decision: record.id(),
                        action_index: index,
                        succeeded: out.succeeded,
                        evidence: out.evidence.clone(),
                    },
                )?;
            }

            self.update_streak(action, &disposition, outcome.as_ref());

            results.push(ActionResult {
                index,
                tier,
                disposition,
                outcome,
            });
        }

        // A violation anywhere in the decision withdraws every grant the
        // decision touched. Shortening the leash needs no evidence and no
        // deliberation.
        if !violations.is_empty() {
            for action in record.actions() {
                let class = ActionClass::of(action);
                self.promotions.revoke(class);
                self.streaks.insert(class, 0);
            }
        }

        Ok(Dispatch {
            results,
            violations,
            approvals_requested,
        })
    }

    fn update_streak(
        &mut self,
        action: &ProposedAction,
        disposition: &ActionDisposition,
        outcome: Option<&Outcome>,
    ) {
        let class = ActionClass::of(action);
        let clean = outcome.map(|o| o.violations.is_empty()).unwrap_or(false);

        match disposition {
            // Only a human saying yes counts toward earning autonomy. An
            // action that was already automatic teaches nothing about whether
            // it should be — the human never saw it.
            ActionDisposition::Approved { succeeded: true } if clean => {
                *self.streaks.entry(class).or_insert(0) += 1;
            }
            ActionDisposition::Executed { .. } => {}
            _ => {
                self.streaks.insert(class, 0);
            }
        }
    }

    /// Classes that have earned enough consecutive clean approvals to be
    /// considered for promotion.
    ///
    /// `brier` is the agent's current calibration; `None` means not yet
    /// measurable. Poor calibration blocks every candidate, because a system
    /// that does not know what it does not know should not be acting
    /// unsupervised, however reliable it has happened to be so far.
    ///
    /// This returns *candidates*, never grants. The actual promotion is
    /// certificate-gated; see `samaritan-cert`.
    pub fn promotion_candidates(
        &self,
        min_streak: u32,
        brier: Option<f64>,
        calibration_ceiling: f64,
    ) -> Vec<ActionClass> {
        let Some(b) = brier else {
            return Vec::new();
        };
        if b > calibration_ceiling {
            return Vec::new();
        }
        self.streaks
            .iter()
            .filter(|(class, streak)| {
                **streak >= min_streak && self.ceiling.permits_promotion(**class)
            })
            .map(|(class, _)| *class)
            .collect()
    }

    /// Grant a class automatic execution.
    ///
    /// Returns whether it took. Callers are expected to have a certificate in
    /// hand; the kernel enforces the ceiling regardless of what the caller
    /// believes.
    pub fn grant(&mut self, class: ActionClass) -> bool {
        self.promotions.grant(class, &self.ceiling)
    }

    /// Withdraw a grant. Always permitted.
    pub fn revoke(&mut self, class: ActionClass) {
        self.promotions.revoke(class);
        self.streaks.insert(class, 0);
    }

    /// Whether self-modification is affordable at the current budget tier.
    pub fn may_self_modify(&self) -> bool {
        self.budget.tier().allows_self_modification()
    }

    pub fn compute_tier(&self) -> ComputeTier {
        self.budget.tier()
    }
}

fn misgrade_violation(claimed: &ProposedAction, observed: &Outcome) -> Violation {
    Violation {
        tag: "action_misgraded".into(),
        detail: format!(
            "claimed {:?}/{:?}/{:?} but observed {:?}/{:?}/{:?}",
            claimed.kind,
            claimed.reversibility,
            claimed.blast_radius,
            observed.observed_kind,
            observed.observed_reversibility,
            observed.observed_blast_radius,
        ),
    }
}

/// An approval gate that refuses everything.
///
/// The correct default for an unattended run: anything the frozen rules did
/// not already make automatic simply does not happen.
pub struct RefuseAll;

impl ApprovalGate for RefuseAll {
    fn ask(&mut self, _request: &ApprovalRequest) -> Result<Approval, samaritan_kernel::GateError> {
        Ok(Approval::Refuse)
    }
}

/// The authority an episode should carry once it has read anything from the
/// environment.
///
/// A convenience so callers do not hand-roll the taint rule and get it subtly
/// wrong in one place.
pub fn taint_after_observation(current: Authority) -> Authority {
    current.least(Authority::Observed)
}
