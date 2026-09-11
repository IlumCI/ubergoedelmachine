//! Tests for the router.
//!
//! Where possible these assert on the *filesystem* rather than on the
//! router's own report: an action that was supposed to be denied is proven
//! denied by the file it did not write. A router that lies about what it did
//! would pass a test that only reads its return value.

use samaritan_dsl::decision::PolicyVersion;
use samaritan_dsl::{
    ActionKind, Authority, BlastRadius, Confidence, DecisionId, DecisionOption, DecisionRecord,
    Digest, Prediction, ProposedAction, Reversibility,
};
use samaritan_exec::{Confinement, Executor};
use samaritan_kernel::{
    ActionClass, Approval, ApprovalGate, ApprovalRequest, AutonomyCeiling, ComputeBudget,
    GateError,
};
use samaritan_ledger::{FixedClock, Ledger};
use samaritan_router::{ActionDisposition, RefuseAll, Router, RouterError};

// ------------------------------------------------------------- fixtures

fn ledger() -> Ledger {
    Ledger::in_memory(Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap()
}

fn sandbox() -> (tempfile::TempDir, Executor) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("sandbox");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("seed.txt"), "seed").unwrap();
    let e = Executor::new(&root, Confinement::PathChecked).unwrap();
    (dir, e)
}

fn write_action(kind: ActionKind, blast: BlastRadius, path: &str) -> ProposedAction {
    ProposedAction {
        kind,
        reversibility: Reversibility::Snapshot,
        blast_radius: blast,
        intent: format!("write {path}"),
        payload: serde_json::json!({"do": "write_file", "path": path, "contents": "written"}),
    }
}

fn record(actions: Vec<ProposedAction>, authority: Authority) -> DecisionRecord {
    DecisionRecord::new(
        DecisionId::new(),
        "a test fails".into(),
        vec![
            DecisionOption {
                summary: "patch it".into(),
                assessment: "small".into(),
            },
            DecisionOption {
                summary: "leave it".into(),
                assessment: "no".into(),
            },
        ],
        0,
        "the patch is small".into(),
        Prediction {
            outcome: "tests pass".into(),
            confidence: Confidence::new(0.8).unwrap(),
        },
        actions,
        PolicyVersion(Digest::ZERO),
        authority,
    )
    .unwrap()
}

/// A gate that answers from a script and counts how often it was asked.
struct Scripted {
    answers: Vec<Result<Approval, GateError>>,
    asked: usize,
}

impl Scripted {
    fn always(a: Approval) -> Self {
        Self {
            answers: vec![Ok(a)],
            asked: 0,
        }
    }
    fn broken() -> Self {
        Self {
            answers: vec![Err(GateError::Unattended)],
            asked: 0,
        }
    }
}

impl ApprovalGate for Scripted {
    fn ask(&mut self, _r: &ApprovalRequest) -> Result<Approval, GateError> {
        let i = self.asked.min(self.answers.len() - 1);
        self.asked += 1;
        match &self.answers[i] {
            Ok(a) => Ok(*a),
            Err(_) => Err(GateError::Unattended),
        }
    }
}

fn router() -> Router {
    Router::new(AutonomyCeiling::closed(), ComputeBudget::new(1_000_000))
}

// ------------------------------------------------------------- the tiers

#[test]
fn a_sandboxed_write_runs_without_asking() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut gate = Scripted::always(Approval::Refuse);
    let root = ex.root().to_path_buf();

    let rec = record(
        vec![write_action(
            ActionKind::Write,
            BlastRadius::Episode,
            "out.txt",
        )],
        Authority::Task,
    );
    let d = router()
        .dispatch(&rec, &mut ex, &mut gate, &mut l)
        .unwrap();

    assert_eq!(
        d.results[0].disposition,
        ActionDisposition::Executed { succeeded: true }
    );
    assert_eq!(gate.asked, 0, "nobody should have been interrupted");
    assert_eq!(d.approvals_requested, 0);
    assert!(root.join("out.txt").exists());
}

#[test]
fn a_denied_action_is_not_executed_and_nobody_is_asked() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut gate = Scripted::always(Approval::Allow);
    let root = ex.root().to_path_buf();

    // GitHistory against the real repo is denied by the frozen rules.
    let action = ProposedAction {
        kind: ActionKind::GitHistory,
        reversibility: Reversibility::Snapshot,
        blast_radius: BlastRadius::Repo,
        intent: "rewrite history".into(),
        payload: serde_json::json!({"do": "write_file", "path": "rewritten.txt", "contents": "x"}),
    };
    let d = router()
        .dispatch(&record(vec![action], Authority::Task), &mut ex, &mut gate, &mut l)
        .unwrap();

    assert_eq!(d.results[0].disposition, ActionDisposition::Denied);
    assert_eq!(gate.asked, 0, "a denial is not a question");
    assert!(
        !root.join("rewritten.txt").exists(),
        "the action ran despite being denied"
    );
}

#[test]
fn a_confirm_action_waits_and_runs_only_on_yes() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let root = ex.root().to_path_buf();

    // A repo-scoped write is Confirm by the frozen rules.
    let rec = record(
        vec![write_action(ActionKind::Write, BlastRadius::Repo, "approved.txt")],
        Authority::Task,
    );

    let mut yes = Scripted::always(Approval::Allow);
    let d = router().dispatch(&rec, &mut ex, &mut yes, &mut l).unwrap();
    assert_eq!(
        d.results[0].disposition,
        ActionDisposition::Approved { succeeded: true }
    );
    assert_eq!(yes.asked, 1);
    assert_eq!(d.approvals_requested, 1);
    assert!(root.join("approved.txt").exists());
}

#[test]
fn a_refused_action_does_not_happen() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let root = ex.root().to_path_buf();
    let mut no = Scripted::always(Approval::Refuse);

    let rec = record(
        vec![write_action(ActionKind::Write, BlastRadius::Repo, "refused.txt")],
        Authority::Task,
    );
    let d = router().dispatch(&rec, &mut ex, &mut no, &mut l).unwrap();

    assert_eq!(d.results[0].disposition, ActionDisposition::Refused);
    assert!(!root.join("refused.txt").exists());
}

#[test]
fn an_unreachable_human_is_a_refusal_not_an_approval() {
    // The single most important line in the crate. If a gate failure read as
    // consent, taking down the approval channel would be the cheapest
    // exploit available to an adversary.
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let root = ex.root().to_path_buf();
    let mut broken = Scripted::broken();

    let rec = record(
        vec![write_action(ActionKind::Write, BlastRadius::Repo, "unattended.txt")],
        Authority::Task,
    );
    let d = router().dispatch(&rec, &mut ex, &mut broken, &mut l).unwrap();

    assert!(matches!(
        d.results[0].disposition,
        ActionDisposition::Unattended { .. }
    ));
    assert!(!d.results[0].disposition.ran());
    assert!(
        !root.join("unattended.txt").exists(),
        "an unreachable human must never mean yes"
    );
}

#[test]
fn the_default_gate_refuses_everything() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let root = ex.root().to_path_buf();

    let rec = record(
        vec![write_action(ActionKind::Write, BlastRadius::Repo, "nope.txt")],
        Authority::Task,
    );
    let d = router()
        .dispatch(&rec, &mut ex, &mut RefuseAll, &mut l)
        .unwrap();
    assert_eq!(d.results[0].disposition, ActionDisposition::Refused);
    assert!(!root.join("nope.txt").exists());
}

// ----------------------------------------------------------- provenance

#[test]
fn adversary_written_reasoning_loses_the_automatic_path() {
    // Same action, same policy, different provenance. Under Task authority it
    // runs unattended; once the reasoning came from something the Deviant
    // could have written, a human is asked.
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let action = write_action(ActionKind::Write, BlastRadius::Episode, "x.txt");

    let mut gate = Scripted::always(Approval::Refuse);
    let trusted = router()
        .dispatch(&record(vec![action.clone()], Authority::Task), &mut ex, &mut gate, &mut l)
        .unwrap();
    assert_eq!(trusted.approvals_requested, 0);

    let mut gate2 = Scripted::always(Approval::Refuse);
    let tainted = router()
        .dispatch(&record(vec![action], Authority::Observed), &mut ex, &mut gate2, &mut l)
        .unwrap();
    assert_eq!(
        tainted.approvals_requested, 1,
        "observed-authority input must not run unattended"
    );
    assert_eq!(tainted.results[0].disposition, ActionDisposition::Refused);
}

// ------------------------------------------------------------ violations

#[test]
fn an_escape_attempt_is_reported_as_a_violation() {
    let (guard, mut ex) = sandbox();
    let mut l = ledger();
    let outside = guard.path().join("escaped.txt");

    let action = write_action(ActionKind::Write, BlastRadius::Episode, "../escaped.txt");
    let d = router()
        .dispatch(&record(vec![action], Authority::Task), &mut ex, &mut RefuseAll, &mut l)
        .unwrap();

    assert!(!d.clean());
    assert_eq!(d.violations[0].tag, "sandbox_escape_attempt");
    assert!(!outside.exists());
}

#[test]
fn a_misreported_action_is_a_violation() {
    // Claims a read, performs a write. Everything downstream trusts these
    // labels, so understating one is charged rather than merely noted.
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let action = ProposedAction {
        kind: ActionKind::Read,
        reversibility: Reversibility::Trivial,
        blast_radius: BlastRadius::Episode,
        intent: "just looking".into(),
        payload: serde_json::json!({"do": "write_file", "path": "sneaky.txt", "contents": "x"}),
    };
    let d = router()
        .dispatch(&record(vec![action], Authority::Task), &mut ex, &mut RefuseAll, &mut l)
        .unwrap();

    assert!(!d.clean());
    assert!(d.violations.iter().any(|v| v.tag == "action_misgraded"));
}

#[test]
fn an_honest_subprocess_is_not_flagged_for_the_sandboxes_own_weakness() {
    // Under PathChecked every subprocess carries a machine-wide blast radius
    // because nothing confines it. That is the sandbox's limitation, not the
    // agent's dishonesty, and a misgrade detector that fires on every single
    // Run detects nothing at all.
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let action = ProposedAction {
        kind: ActionKind::Exec,
        reversibility: Reversibility::Snapshot,
        blast_radius: BlastRadius::Episode,
        intent: "check the git version".into(),
        payload: serde_json::json!({
            "do": "run", "program": "git", "args": ["--version"], "timeout_secs": 30
        }),
    };
    let d = router()
        .dispatch(&record(vec![action], Authority::Task), &mut ex, &mut Scripted::always(Approval::Allow), &mut l)
        .unwrap();

    assert!(
        !d.violations.iter().any(|v| v.tag == "action_misgraded"),
        "got spurious misgrade: {:?}",
        d.violations
    );
}

// ------------------------------------------------------------- autonomy

fn repo_write_class() -> ActionClass {
    ActionClass {
        kind: ActionKind::Write,
        reversibility: Reversibility::Snapshot,
        blast_radius: BlastRadius::Repo,
    }
}

#[test]
fn only_human_approvals_build_a_streak() {
    // An action that was already automatic teaches nothing about whether it
    // should be: the human never saw it.
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut r = router();
    let auto = write_action(ActionKind::Write, BlastRadius::Episode, "a.txt");

    for _ in 0..5 {
        r.dispatch(&record(vec![auto.clone()], Authority::Task), &mut ex, &mut RefuseAll, &mut l)
            .unwrap();
    }
    assert_eq!(
        r.streak(ActionClass::of(&auto)),
        0,
        "automatic executions must not earn further autonomy"
    );
}

#[test]
fn consecutive_clean_approvals_accumulate() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut r = router();
    let mut yes = Scripted::always(Approval::Allow);
    let action = write_action(ActionKind::Write, BlastRadius::Repo, "b.txt");

    for _ in 0..3 {
        r.dispatch(&record(vec![action.clone()], Authority::Task), &mut ex, &mut yes, &mut l)
            .unwrap();
    }
    assert_eq!(r.streak(repo_write_class()), 3);
}

#[test]
fn a_refusal_resets_the_streak() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut r = router();
    let action = write_action(ActionKind::Write, BlastRadius::Repo, "c.txt");

    let mut yes = Scripted::always(Approval::Allow);
    r.dispatch(&record(vec![action.clone()], Authority::Task), &mut ex, &mut yes, &mut l)
        .unwrap();
    assert_eq!(r.streak(repo_write_class()), 1);

    let mut no = Scripted::always(Approval::Refuse);
    r.dispatch(&record(vec![action], Authority::Task), &mut ex, &mut no, &mut l)
        .unwrap();
    assert_eq!(r.streak(repo_write_class()), 0);
}

#[test]
fn a_violation_revokes_immediately_and_without_ceremony() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let class = repo_write_class();
    let mut r = Router::new(
        AutonomyCeiling::allowing([class]),
        ComputeBudget::new(1_000_000),
    );
    assert!(r.grant(class));
    assert!(r.promotions().holds(class));

    // An escape attempt in the same class.
    let bad = write_action(ActionKind::Write, BlastRadius::Repo, "../out.txt");
    let d = r
        .dispatch(&record(vec![bad], Authority::Task), &mut ex, &mut RefuseAll, &mut l)
        .unwrap();

    assert!(!d.clean());
    assert!(
        !r.promotions().holds(class),
        "a violation must withdraw the grant on the spot"
    );
    assert_eq!(r.streak(class), 0);
}

#[test]
fn poor_calibration_blocks_every_promotion_candidate() {
    // A system that does not know what it does not know should not be acting
    // unsupervised, however reliable it has happened to be so far.
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let class = repo_write_class();
    let mut r = Router::new(
        AutonomyCeiling::allowing([class]),
        ComputeBudget::new(1_000_000),
    );
    let mut yes = Scripted::always(Approval::Allow);
    let action = write_action(ActionKind::Write, BlastRadius::Repo, "d.txt");
    for _ in 0..5 {
        r.dispatch(&record(vec![action.clone()], Authority::Task), &mut ex, &mut yes, &mut l)
            .unwrap();
    }
    assert_eq!(r.streak(class), 5);

    assert!(
        r.promotion_candidates(3, Some(0.40), 0.20).is_empty(),
        "an overconfident agent earns nothing"
    );
    assert!(
        r.promotion_candidates(3, None, 0.20).is_empty(),
        "unmeasured calibration is not good calibration"
    );
    assert_eq!(
        r.promotion_candidates(3, Some(0.10), 0.20),
        vec![class],
        "well calibrated and a long enough streak"
    );
}

#[test]
fn a_class_outside_the_ceiling_is_never_a_candidate() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut r = router(); // closed ceiling
    let mut yes = Scripted::always(Approval::Allow);
    let action = write_action(ActionKind::Write, BlastRadius::Repo, "e.txt");
    for _ in 0..5 {
        r.dispatch(&record(vec![action.clone()], Authority::Task), &mut ex, &mut yes, &mut l)
            .unwrap();
    }
    assert!(r.promotion_candidates(3, Some(0.05), 0.20).is_empty());
}

// --------------------------------------------------------------- budget

#[test]
fn an_exhausted_budget_stops_work_entirely() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut r = Router::new(AutonomyCeiling::closed(), ComputeBudget::new(100));
    r.spend(100);

    let rec = record(
        vec![write_action(ActionKind::Write, BlastRadius::Episode, "f.txt")],
        Authority::Task,
    );
    assert!(matches!(
        r.dispatch(&rec, &mut ex, &mut RefuseAll, &mut l),
        Err(RouterError::ComputeExhausted)
    ));
    assert!(!ex.root().join("f.txt").exists());
}

#[test]
fn self_modification_stops_while_task_work_continues() {
    let mut r = Router::new(AutonomyCeiling::closed(), ComputeBudget::new(1_000));
    assert!(r.may_self_modify());
    r.spend(850); // 15% left
    assert!(!r.may_self_modify(), "speculation stops first");
    assert!(r.compute_tier().allows_work(), "task work carries on");
}

// --------------------------------------------------------------- ledger

#[test]
fn the_whole_dispatch_is_written_down_and_verifies() {
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut yes = Scripted::always(Approval::Allow);
    let rec = record(
        vec![
            write_action(ActionKind::Write, BlastRadius::Episode, "auto.txt"),
            write_action(ActionKind::Write, BlastRadius::Repo, "asked.txt"),
        ],
        Authority::Task,
    );
    router().dispatch(&rec, &mut ex, &mut yes, &mut l).unwrap();

    assert_eq!(l.by_kind("decision_recorded").unwrap().len(), 1);
    assert_eq!(l.by_kind("action_routed").unwrap().len(), 2);
    assert_eq!(l.by_kind("approval_resolved").unwrap().len(), 1);
    assert_eq!(l.by_kind("action_executed").unwrap().len(), 2);
    l.verify().unwrap();
}

#[test]
fn a_refusal_is_recorded_even_though_nothing_ran() {
    // An action that did not happen still has to be visible: the friction it
    // cost and the fact a human said no are both signal.
    let (_g, mut ex) = sandbox();
    let mut l = ledger();
    let mut no = Scripted::always(Approval::Refuse);
    let rec = record(
        vec![write_action(ActionKind::Write, BlastRadius::Repo, "g.txt")],
        Authority::Task,
    );
    router().dispatch(&rec, &mut ex, &mut no, &mut l).unwrap();

    assert_eq!(l.by_kind("approval_resolved").unwrap().len(), 1);
    assert_eq!(
        l.by_kind("action_executed").unwrap().len(),
        0,
        "nothing executed, so nothing claims to have"
    );
}
