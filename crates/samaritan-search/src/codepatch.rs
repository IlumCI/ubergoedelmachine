//! Level 3: the search rewriting its own source.
//!
//! This is the genuinely Gödelian move — the rewriter rewriting the rewriter —
//! and it is the one place in the system where a mistake is not recoverable by
//! rolling back a struct in memory. So it is the most heavily gated thing here,
//! and the gate is ordered deliberately.
//!
//! A patch must clear **four** checks, in this order, and the order is a safety
//! property rather than an optimisation:
//!
//! 1. **Admission screening.** Frozen paths are refused *before the patch is
//!    ever applied or built*. A diff that reaches `samaritan-kernel` or
//!    `samaritan-cert` must never be compiled, let alone run: building it would
//!    execute its build scripts and run its tests, which is exactly the
//!    foothold a patch aimed at the frozen core would want.
//! 2. **It builds.** In a scratch tree, never the live one.
//! 3. **Its tests pass.** The whole suite, not the tests the patch ships with —
//!    a patch that deletes the tests that constrain it must not pass.
//! 4. **A human approves.** Last, mandatory, and never inferred. Building and
//!    passing is *necessary* and explicitly not *sufficient*: the machine can
//!    write a patch that compiles, passes, and is still a terrible idea, and
//!    the only thing that reliably catches that is a person reading it.
//!
//! The human is asked last so that nobody is ever asked to review a patch that
//! does not compile. That is a courtesy to the reviewer, and courtesy to the
//! reviewer is a safety property: a gate that wastes attention gets clicked
//! through.
//!
//! The effectful half — copying the tree, applying the diff, running cargo — is
//! behind [`PatchVerifier`] so the decision logic here is testable without
//! compiling anything, the same seam the domain uses for its evaluator.

use samaritan_dsl::Mutation;
use samaritan_kernel::{Admission, Approval, Refusal, FROZEN_PATHS};
use serde::{Deserialize, Serialize};

/// What a proposer is asked to improve, and what it is shown to do it.
///
/// Deliberately narrow. A proposer sees one file's source and a goal, and is
/// asked for a diff to that file. It is not handed the whole tree: a patch that
/// spans crates is harder to review, harder to verify in isolation, and far
/// likelier to be the kind of sweeping change that should be a human's design
/// decision rather than the machine's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchContext {
    /// What the patch should achieve, in a sentence.
    pub goal: String,
    /// Repo-relative path of the file to patch. Must not be frozen — see
    /// [`PatchContext::targets_frozen`].
    pub target_path: String,
    /// The file's current contents, so a proposer can write a diff that applies.
    pub current_source: String,
}

impl PatchContext {
    /// Whether the target sits in the frozen core. A proposer should never be
    /// *pointed* at one — the gate would refuse the patch anyway, but asking a
    /// model to rewrite the kernel is a request that should not be made in the
    /// first place, not merely one that fails late.
    pub fn targets_frozen(&self) -> bool {
        let p = self.target_path.replace('\\', "/");
        FROZEN_PATHS.iter().any(|f| p.starts_with(*f))
    }
}

/// Proposes a patch to the search's own source. The level-3 counterpart of the
/// Deviant's [`crate`]-external `Attacker`: a trait, so the loop can be driven
/// by a scripted proposer in a test as well as by a live model, and so the part
/// that must be right (the gate) is never entangled with the part that is
/// expensive and non-deterministic (generation).
pub trait PatchProposer {
    /// Propose a patch, or `None` if it has nothing to offer this round. The
    /// returned mutation is always a [`Mutation::CodePatch`]; anything else is a
    /// bug in the proposer, and the gate will reject it as not-a-code-patch.
    fn propose(&mut self, ctx: &PatchContext) -> Option<Mutation>;
}

/// What the verifier found when it tried the patch in a scratch tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchReport {
    pub builds: bool,
    /// Meaningless unless `builds`; a tree that does not compile has no tests.
    pub tests_pass: bool,
    /// Compiler or test output, for the human who has to judge it.
    pub detail: String,
}

impl PatchReport {
    pub fn failed_to_build(detail: impl Into<String>) -> Self {
        Self { builds: false, tests_pass: false, detail: detail.into() }
    }
    pub fn tests_failed(detail: impl Into<String>) -> Self {
        Self { builds: true, tests_pass: false, detail: detail.into() }
    }
    pub fn passed(detail: impl Into<String>) -> Self {
        Self { builds: true, tests_pass: true, detail: detail.into() }
    }
}

/// Applies a candidate patch somewhere that is not the live tree, and reports
/// whether it builds and passes.
///
/// Implementations must never write to the working tree the harness is running
/// from. The gate cannot enforce that — it is a property of the implementation —
/// which is why it is stated here as the contract.
pub trait PatchVerifier {
    fn verify(&mut self, unified_diff: &str) -> PatchReport;
}

/// Why a level-3 patch was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum PatchRefusal {
    /// The frozen rules refused it. It was never built.
    Screened(Refusal),
    /// Not a level-3 mutation at all.
    NotACodePatch,
    DoesNotBuild { detail: String },
    TestsFail { detail: String },
    /// It builds and passes, and no human has said yes. This is a refusal, not
    /// a pending state: the caller must ask and come back.
    AwaitingHuman,
    HumanRefused,
}

impl std::fmt::Display for PatchRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatchRefusal::Screened(r) => write!(f, "refused by the frozen rules: {r}"),
            PatchRefusal::NotACodePatch => f.write_str("not a code patch"),
            PatchRefusal::DoesNotBuild { detail } => write!(f, "does not build: {detail}"),
            PatchRefusal::TestsFail { detail } => write!(f, "tests fail: {detail}"),
            PatchRefusal::AwaitingHuman => {
                f.write_str("builds and passes, but no human has approved it")
            }
            PatchRefusal::HumanRefused => f.write_str("a human refused it"),
        }
    }
}

/// A patch that cleared every gate. Only this grants the right to write to the
/// tree, and it cannot be constructed any other way.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedPatch {
    pub unified_diff: String,
    pub report: PatchReport,
}

/// Run the level-3 gate.
///
/// `human` is the answer a person actually gave, obtained through
/// [`samaritan_kernel::ApprovalGate`]. Passing `Approval::Allow` without having
/// asked is the one way to defeat this gate, which is why the parameter is the
/// answer rather than a callback the caller could stub out.
pub fn gate_code_patch(
    mutation: &Mutation,
    admission: &Admission,
    verifier: &mut dyn PatchVerifier,
    human: Approval,
) -> Result<AcceptedPatch, PatchRefusal> {
    let Mutation::CodePatch { unified_diff } = mutation else {
        return Err(PatchRefusal::NotACodePatch);
    };

    // 1. Screen first. A patch at the frozen core is never compiled.
    admission.admit(mutation).map_err(PatchRefusal::Screened)?;

    // 2 and 3. Build and test in a scratch tree.
    let report = verifier.verify(unified_diff);
    if !report.builds {
        return Err(PatchRefusal::DoesNotBuild { detail: report.detail });
    }
    if !report.tests_pass {
        return Err(PatchRefusal::TestsFail { detail: report.detail });
    }

    // 4. The human, last and mandatory.
    match human {
        Approval::Refuse => Err(PatchRefusal::HumanRefused),
        Approval::Allow => Ok(AcceptedPatch {
            unified_diff: unified_diff.clone(),
            report,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records whether it was ever asked to build, so a test can prove a
    /// screened-out patch was never compiled.
    struct Spy {
        report: PatchReport,
        calls: usize,
    }
    impl Spy {
        fn new(report: PatchReport) -> Self {
            Self { report, calls: 0 }
        }
    }
    impl PatchVerifier for Spy {
        fn verify(&mut self, _diff: &str) -> PatchReport {
            self.calls += 1;
            self.report.clone()
        }
    }

    fn patch(diff: &str) -> Mutation {
        Mutation::CodePatch { unified_diff: diff.into() }
    }

    /// A patch to the search itself — level 3's legitimate target.
    fn own_source() -> Mutation {
        patch("--- a/crates/samaritan-search/src/lib.rs\n+++ b/crates/samaritan-search/src/lib.rs\n")
    }

    fn level3() -> Admission {
        Admission::new(3)
    }

    #[test]
    fn a_patch_that_builds_passes_and_is_approved_is_accepted() {
        let mut v = Spy::new(PatchReport::passed("ok"));
        let got = gate_code_patch(&own_source(), &level3(), &mut v, Approval::Allow);
        assert!(got.is_ok(), "{got:?}");
    }

    #[test]
    fn building_and_passing_is_not_enough_without_a_human() {
        // The property the whole level exists to preserve. A patch that
        // compiles and goes green is *necessary*, never sufficient.
        let mut v = Spy::new(PatchReport::passed("ok"));
        assert_eq!(
            gate_code_patch(&own_source(), &level3(), &mut v, Approval::Refuse),
            Err(PatchRefusal::HumanRefused)
        );
    }

    #[test]
    fn a_frozen_core_patch_is_refused_without_ever_being_built() {
        // Ordering as a safety property: building a patch runs its build
        // scripts and its tests, so a diff aimed at the kernel must be turned
        // away before the compiler ever sees it.
        let mut v = Spy::new(PatchReport::passed("ok"));
        let kernel = patch(
            "--- a/crates/samaritan-kernel/src/admission.rs\n\
             +++ b/crates/samaritan-kernel/src/admission.rs\n",
        );
        let got = gate_code_patch(&kernel, &level3(), &mut v, Approval::Allow);
        assert!(matches!(got, Err(PatchRefusal::Screened(_))), "{got:?}");
        assert_eq!(v.calls, 0, "a frozen-core patch must never be compiled");
    }

    #[test]
    fn the_certificate_is_as_unpatchable_as_the_kernel() {
        let mut v = Spy::new(PatchReport::passed("ok"));
        let cert = patch(
            "--- a/crates/samaritan-cert/src/martingale.rs\n\
             +++ b/crates/samaritan-cert/src/martingale.rs\n",
        );
        let got = gate_code_patch(&cert, &level3(), &mut v, Approval::Allow);
        assert!(matches!(got, Err(PatchRefusal::Screened(_))), "{got:?}");
        assert_eq!(v.calls, 0);
    }

    #[test]
    fn a_patch_that_does_not_build_is_refused_before_the_human_is_asked() {
        let mut v = Spy::new(PatchReport::failed_to_build("E0308"));
        let got = gate_code_patch(&own_source(), &level3(), &mut v, Approval::Allow);
        assert!(matches!(got, Err(PatchRefusal::DoesNotBuild { .. })), "{got:?}");
    }

    #[test]
    fn a_patch_whose_tests_fail_is_refused() {
        let mut v = Spy::new(PatchReport::tests_failed("2 failed"));
        let got = gate_code_patch(&own_source(), &level3(), &mut v, Approval::Allow);
        assert!(matches!(got, Err(PatchRefusal::TestsFail { .. })), "{got:?}");
    }

    #[test]
    fn level_three_being_disabled_refuses_before_anything_runs() {
        // The frozen admission gate is upstream of the whole ladder.
        let mut v = Spy::new(PatchReport::passed("ok"));
        let got = gate_code_patch(&own_source(), &Admission::new(2), &mut v, Approval::Allow);
        assert!(matches!(got, Err(PatchRefusal::Screened(_))), "{got:?}");
        assert_eq!(v.calls, 0);
    }

    #[test]
    fn a_non_code_patch_is_not_this_gate_s_business() {
        let mut v = Spy::new(PatchReport::passed("ok"));
        let m = Mutation::LessonAdd { text: "x".into() };
        assert_eq!(
            gate_code_patch(&m, &level3(), &mut v, Approval::Allow),
            Err(PatchRefusal::NotACodePatch)
        );
    }

    #[test]
    fn an_unreadable_diff_is_refused_rather_than_admitted() {
        // "I could not tell what this changes" must never resolve to "so let it
        // through" — the screen refuses a diff with no recognisable headers.
        let mut v = Spy::new(PatchReport::passed("ok"));
        let got = gate_code_patch(&patch("not a diff at all"), &level3(), &mut v, Approval::Allow);
        assert!(matches!(got, Err(PatchRefusal::Screened(_))), "{got:?}");
        assert_eq!(v.calls, 0);
    }

    /// A proposer that always returns the same prepared patch.
    struct FixedProposer(Mutation);
    impl PatchProposer for FixedProposer {
        fn propose(&mut self, _ctx: &PatchContext) -> Option<Mutation> {
            Some(self.0.clone())
        }
    }

    fn context(target: &str) -> PatchContext {
        PatchContext {
            goal: "make the search a little faster".into(),
            target_path: target.into(),
            current_source: "fn main() {}\n".into(),
        }
    }

    #[test]
    fn a_context_pointed_at_the_search_is_not_frozen_but_the_kernel_is() {
        assert!(!context("crates/samaritan-search/src/lib.rs").targets_frozen());
        assert!(context("crates/samaritan-kernel/src/admission.rs").targets_frozen());
        assert!(context("crates/samaritan-cert/src/martingale.rs").targets_frozen());
        // Backslash spelling must not evade the check.
        assert!(context("crates\\samaritan-kernel\\src\\x.rs").targets_frozen());
    }

    #[test]
    fn a_proposer_driving_the_gate_still_needs_a_human() {
        // End to end at the trait level: a proposer proposes, the gate decides,
        // and even a clean patch waits on a person.
        let mut proposer = FixedProposer(own_source());
        let mut v = Spy::new(PatchReport::passed("ok"));
        let ctx = context("crates/samaritan-search/src/lib.rs");
        let proposed = proposer.propose(&ctx).expect("a proposal");
        assert!(gate_code_patch(&proposed, &level3(), &mut v, Approval::Refuse).is_err());
        assert!(gate_code_patch(&proposed, &level3(), &mut v, Approval::Allow).is_ok());
    }

    #[test]
    fn a_proposer_aimed_at_the_frozen_core_is_screened_before_building() {
        // Even if a proposer hands back a kernel patch, the gate turns it away
        // without compiling it.
        let kernel = patch(
            "--- a/crates/samaritan-kernel/src/tier.rs\n\
             +++ b/crates/samaritan-kernel/src/tier.rs\n",
        );
        let mut proposer = FixedProposer(kernel);
        let mut v = Spy::new(PatchReport::passed("ok"));
        let proposed = proposer.propose(&context("crates/samaritan-search/src/lib.rs")).unwrap();
        assert!(matches!(
            gate_code_patch(&proposed, &level3(), &mut v, Approval::Allow),
            Err(PatchRefusal::Screened(_))
        ));
        assert_eq!(v.calls, 0);
    }
}
