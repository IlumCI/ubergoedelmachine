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
use samaritan_kernel::{Admission, Approval, Refusal};
use serde::{Deserialize, Serialize};

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
}
