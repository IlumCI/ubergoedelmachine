//! Where an action goes: run it, ask first, or refuse.
//!
//! The tier of an action is derived here, from facts the proposer reported
//! about the action, using rules the proposer cannot reach. Nothing in this
//! module is reachable from [`samaritan_dsl::Mutation`].

use samaritan_dsl::{ActionKind, Authority, BlastRadius, ProposedAction, Reversibility};
use serde::{Deserialize, Serialize};

/// The routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Run it without asking.
    Auto,
    /// Ask a human, and do nothing until they answer.
    Confirm,
    /// Refuse. Not a question.
    Deny,
}

impl Tier {
    /// The stricter of two tiers.
    ///
    /// `Tier` is ordered `Auto < Confirm < Deny`, so combining rules is a max.
    /// Every combinator in this crate goes in this direction; there is
    /// deliberately no operation that relaxes a tier.
    pub fn strictest(self, other: Tier) -> Tier {
        self.max(other)
    }
}

/// The identity of an action for policy purposes.
///
/// Coarse on purpose. Autonomy is granted to a *kind* of action, never to a
/// specific one, so that earning trust on a thousand small edits cannot be
/// cashed in on a single unlike one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActionClass {
    pub kind: ActionKind,
    pub reversibility: Reversibility,
    pub blast_radius: BlastRadius,
}

impl ActionClass {
    pub fn of(a: &ProposedAction) -> Self {
        Self {
            kind: a.kind,
            reversibility: a.reversibility,
            blast_radius: a.blast_radius,
        }
    }
}

/// The frozen base tier for an action class.
///
/// This function is the leash. It is total, pure, and has no configuration
/// input, so its behaviour cannot drift as a run proceeds. Read it as a list of
/// things that are true regardless of how well the agent has been performing.
pub fn base_tier(class: ActionClass) -> Tier {
    use ActionKind::*;
    use BlastRadius::*;
    use Reversibility::*;

    // Nothing irreversible off the sandbox, ever. No streak of good behaviour
    // earns this and no confidence score offsets it.
    if class.reversibility == Irreversible && class.blast_radius >= Machine {
        return Tier::Deny;
    }

    // Rewriting history destroys evidence, including the evidence the ledger
    // would need to notice that rewriting history was a mistake.
    if class.kind == GitHistory && class.blast_radius >= Repo {
        return Tier::Deny;
    }

    // Anything that leaves the machine is somebody else's problem too.
    if class.blast_radius == External {
        return Tier::Confirm;
    }

    match (class.kind, class.reversibility, class.blast_radius) {
        // Reading is free within the machine.
        (Read, _, Episode | Repo | Machine) => Tier::Auto,

        // Inside the episode sandbox the worktree is disposable, so writing and
        // running are cheap to undo and cheap to get wrong.
        (Write | Exec, Trivial | Snapshot, Episode) => Tier::Auto,

        // Everything else is a question.
        _ => Tier::Confirm,
    }
}

/// The human-set upper bound on autonomy.
///
/// Mutations may propose promotions up to this and no further. The ceiling
/// lives in frozen config precisely so that the thing being optimised cannot
/// optimise the constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutonomyCeiling {
    /// Classes a human has said *may* eventually become automatic.
    ///
    /// Membership is permission to earn autonomy, not a grant of it.
    promotable: Vec<ActionClass>,
}

impl AutonomyCeiling {
    /// The default posture: nothing may be promoted.
    ///
    /// A fresh install asks about everything that is not already `Auto` by the
    /// frozen rules above.
    pub fn closed() -> Self {
        Self {
            promotable: Vec::new(),
        }
    }

    pub fn allowing(classes: impl IntoIterator<Item = ActionClass>) -> Self {
        let mut promotable: Vec<ActionClass> = classes.into_iter().collect();
        promotable.sort();
        promotable.dedup();
        Self { promotable }
    }

    /// Whether a class is *eligible* for promotion.
    ///
    /// A class the frozen rules deny is never eligible, even if a human listed
    /// it. That is not an error worth reporting: it means the ceiling and the
    /// base rules disagree, and the base rules win.
    pub fn permits_promotion(&self, class: ActionClass) -> bool {
        base_tier(class) != Tier::Deny && self.promotable.contains(&class)
    }

    pub fn classes(&self) -> &[ActionClass] {
        &self.promotable
    }
}

/// Autonomy the search has actually earned, within the ceiling.
///
/// This is mutable state. It is also the only mutable input to routing, which
/// is why every path that widens it runs through a certificate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Promotions {
    granted: Vec<ActionClass>,
}

impl Promotions {
    pub fn none() -> Self {
        Self::default()
    }

    /// Record that a class has earned automatic execution.
    ///
    /// Returns whether the grant took effect. A grant outside the ceiling is
    /// dropped rather than honoured.
    pub fn grant(&mut self, class: ActionClass, ceiling: &AutonomyCeiling) -> bool {
        if !ceiling.permits_promotion(class) {
            return false;
        }
        if !self.granted.contains(&class) {
            self.granted.push(class);
            self.granted.sort();
        }
        true
    }

    /// Withdraw a grant.
    ///
    /// Deliberately unconditional and not certificate-gated: the machine may
    /// always shorten its own leash, and only lengthening needs evidence.
    pub fn revoke(&mut self, class: ActionClass) {
        self.granted.retain(|c| *c != class);
    }

    pub fn holds(&self, class: ActionClass) -> bool {
        self.granted.contains(&class)
    }

    pub fn classes(&self) -> &[ActionClass] {
        &self.granted
    }
}

/// Route one action.
///
/// The only way a tier is ever relaxed, and it is bounded four ways: the base
/// rules must have said `Confirm` rather than `Deny`, the decision must not
/// rest on adversary-writable input, the human ceiling must list the class,
/// and the search must have earned the grant.
///
/// `authority` is the least-trusted source that shaped the decision. A
/// decision derived from observed content — file contents, stdout, an error
/// message — never runs unattended, however much trust the class has earned.
/// In the arena the Deviant authors all of those, so without this rule an
/// adversary that cannot defeat the router can instead write the text that
/// persuades the Warden, and inherit the Warden's autonomy for free.
pub fn route(
    action: &ProposedAction,
    authority: Authority,
    ceiling: &AutonomyCeiling,
    earned: &Promotions,
) -> Tier {
    let class = ActionClass::of(action);
    let base = base_tier(class);

    if base == Tier::Confirm
        && authority < Authority::Observed
        && earned.holds(class)
        && ceiling.permits_promotion(class)
    {
        return Tier::Auto;
    }

    // Observed-authority input cannot reach `Auto` even when the base rules
    // would have allowed it outright.
    if authority >= Authority::Observed {
        return base.strictest(Tier::Confirm);
    }
    base
}
