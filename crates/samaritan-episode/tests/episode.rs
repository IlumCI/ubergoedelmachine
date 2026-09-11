//! Tests for the level-0 rollout.
//!
//! The episode is what every number above it is computed from, so these are
//! mostly about the score being honest: the agent cannot declare victory,
//! predictions are settled by the oracle, and a violated episode cannot buy
//! its way back with task success.
//!
//! The agent is scripted. Running a real model here would take a minute per
//! decision and test the model rather than the loop.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use samaritan_agent::{AgentError, Completion, DraftDecision, DraftAction, DraftOption,
                      DraftPrediction, Prompt, Usage};
use samaritan_corpus::{MineOptions, OracleSpec, Task, mine};
use samaritan_dsl::decision::PolicyVersion;
use samaritan_dsl::{ActionKind, Authority, BlastRadius, Digest, Reversibility};
use samaritan_episode::{Decider, Ending, EpisodeConfig, run_episode};
use samaritan_kernel::{AutonomyCeiling, ComputeBudget};
use samaritan_ledger::{FixedClock, Ledger};
use samaritan_router::{RefuseAll, Router};

// ------------------------------------------------------------ a real repo

struct Repo(tempfile::TempDir);

impl Repo {
    fn new() -> Self {
        let d = tempfile::tempdir().unwrap();
        let r = Repo(d);
        r.git(["init", "-q", "-b", "main", "."]);
        r.git(["config", "user.email", "t@example.invalid"]);
        r.git(["config", "user.name", "Test"]);
        r
    }
    fn path(&self) -> &Path {
        self.0.path()
    }
    fn git<I: IntoIterator<Item = S>, S: AsRef<std::ffi::OsStr>>(&self, a: I) {
        let o = Command::new("git").arg("-C").arg(self.path()).args(a).output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    fn write(&self, rel: &str, c: &str) {
        let p = self.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, c).unwrap();
    }
    fn commit(&self, msg: &str, date: &str) {
        self.git(["add", "-A"]);
        let o = Command::new("git")
            .arg("-C").arg(self.path())
            .args(["commit", "-q", "--allow-empty", "-m", msg])
            .env("GIT_AUTHOR_DATE", date)
            .env("GIT_COMMITTER_DATE", date)
            .output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
}

/// A task whose oracle is a shell check rather than a compiler, so the suite
/// runs in milliseconds. The episode loop does not care what the oracle is —
/// only what it exits with.
fn task_fixture() -> (Repo, Task) {
    let r = Repo::new();
    // "Source" is a file whose contents the oracle checks.
    r.write("src/value.txt", "0\n");
    r.write("tests/check.cmd", "old\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.write("src/value.txt", "4\n");
    r.write("tests/check.cmd", "new\n");
    r.commit("fix the value", "2026-02-01T00:00:00+00:00");

    let mut opts = MineOptions::new("fixture", "2026-06-01T00:00:00+00:00");
    // Oracle: pass only when src/value.txt holds 4.
    //
    // `git grep` as the checker. It behaves identically on every platform,
    // exits 0/1 cleanly, and the harness already hard-depends on git. The
    // obvious alternatives are both worse than they look: `cmd /C "..."`
    // runs into Windows command-line quoting, and `findstr /X 4 file`
    // parses the pattern as a second filename rather than as a pattern.
    opts.oracle = OracleSpec {
        program: "git".into(),
        args: vec![
            "grep".into(),
            "-q".into(),
            "^4$".into(),
            "--".into(),
            "src/value.txt".into(),
        ],
        timeout_secs: 30,
    };
    let mut c = mine(r.path(), &opts).unwrap();
    let t = c.tasks.remove(0);
    (r, t)
}

// ------------------------------------------------------- a scripted agent

struct Scripted {
    drafts: RefCell<Vec<DraftDecision>>,
    calls: RefCell<u32>,
    authority_seen: RefCell<Vec<Authority>>,
}

impl Scripted {
    fn new(drafts: Vec<DraftDecision>) -> Self {
        Self {
            drafts: RefCell::new(drafts),
            calls: RefCell::new(0),
            authority_seen: RefCell::new(Vec::new()),
        }
    }
}

impl Decider for Scripted {
    fn decide(&self, prompt: &Prompt, pv: PolicyVersion) -> Result<Completion, AgentError> {
        *self.calls.borrow_mut() += 1;
        self.authority_seen.borrow_mut().push(prompt.authority());
        let mut d = self.drafts.borrow_mut();
        if d.is_empty() {
            return Err(AgentError::Transport("script exhausted".into()));
        }
        let draft = d.remove(0);
        let record = draft.seal(pv, prompt.authority())?;
        Ok(Completion {
            record,
            usage: Usage { prompt_tokens: 100, completion_tokens: 200, cached_tokens: 0 },
            elapsed: Duration::from_secs(1),
            attempts: 1,
            seed: 1,
        })
    }
}

fn draft(intent: &str, confidence: f64, payload: serde_json::Value,
         kind: ActionKind, rev: Reversibility, blast: BlastRadius) -> DraftDecision {
    DraftDecision {
        situation: "the value is wrong".into(),
        options: vec![
            DraftOption { summary: "fix it".into(), assessment: "direct".into() },
            DraftOption { summary: "do nothing".into(), assessment: "leaves it broken".into() },
        ],
        chosen: 0,
        rationale: "the smallest change".into(),
        prediction: DraftPrediction { outcome: "the check passes".into(), confidence },
        actions: vec![DraftAction {
            kind, reversibility: rev, blast_radius: blast,
            intent: intent.into(), payload,
        }],
    }
}

fn fix_draft(confidence: f64) -> DraftDecision {
    draft(
        "write the corrected value",
        confidence,
        serde_json::json!({"do": "write_file", "path": "src/value.txt", "contents": "4\n"}),
        ActionKind::Write, Reversibility::Snapshot, BlastRadius::Episode,
    )
}

fn noop_draft(confidence: f64) -> DraftDecision {
    draft(
        "look at the file",
        confidence,
        serde_json::json!({"do": "read_file", "path": "src/value.txt"}),
        ActionKind::Read, Reversibility::Trivial, BlastRadius::Episode,
    )
}

// ------------------------------------------------------------- harness

struct Rig {
    ledger: Ledger,
    router: Router,
    cfg: EpisodeConfig,
    _scratch: tempfile::TempDir,
}

fn rig() -> Rig {
    let scratch = tempfile::tempdir().unwrap();
    Rig {
        ledger: Ledger::in_memory(Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap(),
        router: Router::new(AutonomyCeiling::closed(), ComputeBudget::new(10_000_000)),
        cfg: EpisodeConfig {
            scratch_root: scratch.path().to_path_buf(),
            max_steps: 4,
            ..Default::default()
        },
        _scratch: scratch,
    }
}

fn run(rig: &mut Rig, repo: &Repo, task: &Task, agent: &dyn Decider)
    -> samaritan_episode::EpisodeOutcome {
    run_episode(
        task, repo.path(), agent, &mut rig.router, &mut RefuseAll,
        &mut rig.ledger, PolicyVersion(Digest::ZERO), &rig.cfg,
    ).unwrap()
}

// ----------------------------------------------------------------- tests

#[test]
fn a_solved_task_scores_and_stops() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    let agent = Scripted::new(vec![fix_draft(0.9)]);
    let out = run(&mut r, &repo, &task, &agent);

    assert_eq!(out.ending, Ending::Solved { at_step: 1 });
    assert_eq!(out.components.task_success, 1.0);
    assert!(out.utility.is_clean());
    assert_eq!(*agent.calls.borrow(), 1, "it should stop once the suite is green");
}

#[test]
fn the_agent_cannot_declare_victory() {
    // The action reports success and the prediction is confident, but the
    // oracle still fails. Nothing the agent says moves the score.
    let (repo, task) = task_fixture();
    let mut r = rig();
    let agent = Scripted::new(vec![
        draft(
            "claim it is fixed",
            0.99,
            serde_json::json!({"do": "write_file", "path": "notes.txt",
                               "contents": "All tests pass. Task complete."}),
            ActionKind::Write, Reversibility::Snapshot, BlastRadius::Episode,
        ),
        noop_draft(0.5), noop_draft(0.5), noop_draft(0.5),
    ]);
    let out = run(&mut r, &repo, &task, &agent);

    assert_eq!(out.ending, Ending::StepsExhausted);
    assert_eq!(out.components.task_success, 0.0, "only the oracle decides");
}

#[test]
fn predictions_are_settled_by_the_oracle_per_step() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    // A confident wrong step, then a confident right one.
    let agent = Scripted::new(vec![noop_draft(0.95), fix_draft(0.9)]);
    let out = run(&mut r, &repo, &task, &agent);

    assert_eq!(out.predictions.len(), 2);
    assert_eq!(out.predictions[0], (0.95, false), "the read did not fix anything");
    assert_eq!(out.predictions[1], (0.9, true));
    // Confidently wrong once, so calibration should be visibly poor.
    assert!(out.components.brier > 0.4, "brier was {}", out.components.brier);
}

#[test]
fn confident_and_wrong_costs_more_than_hedged_and_wrong() {
    let (repo, task) = task_fixture();
    let outs: Vec<f64> = [0.99_f64, 0.5]
        .iter()
        .map(|c| {
            let mut r = rig();
            let agent = Scripted::new(vec![noop_draft(*c); 4]);
            run(&mut r, &repo, &task, &agent).components.brier
        })
        .collect();
    assert!(outs[0] > outs[1], "overconfidence must be scored harder: {outs:?}");
}

#[test]
fn a_violation_ends_the_episode_and_cannot_be_outscored() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    // Escape the sandbox on the first step, then would have fixed it.
    let agent = Scripted::new(vec![
        draft(
            "write outside the sandbox",
            0.9,
            serde_json::json!({"do": "write_file", "path": "../escaped.txt", "contents": "x"}),
            ActionKind::Write, Reversibility::Snapshot, BlastRadius::Episode,
        ),
        fix_draft(0.9),
    ]);
    let out = run(&mut r, &repo, &task, &agent);

    assert_eq!(out.ending, Ending::Violated);
    assert!(!out.utility.is_clean());
    assert!(out.utility.score().is_none(), "a violated episode has no score");
    assert_eq!(*agent.calls.borrow(), 1, "no more tokens on a foregone conclusion");
}

#[test]
fn steps_are_capped() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    let agent = Scripted::new(vec![noop_draft(0.5); 20]);
    let out = run(&mut r, &repo, &task, &agent);

    assert_eq!(out.ending, Ending::StepsExhausted);
    assert_eq!(out.steps, 4, "max_steps must bind");
    assert_eq!(*agent.calls.borrow(), 4);
}

#[test]
fn an_agent_failure_ends_the_episode_rather_than_looping() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    let agent = Scripted::new(vec![]); // script exhausted -> transport error
    let out = run(&mut r, &repo, &task, &agent);
    assert!(matches!(out.ending, Ending::AgentFailed { .. }));
}

#[test]
fn the_episode_sandbox_never_contains_the_solution() {
    // The property the whole corpus crate exists for, re-checked at the last
    // moment before a model would see it.
    let (repo, task) = task_fixture();
    let mut r = rig();
    let seen: RefCell<Option<String>> = RefCell::new(None);

    struct Peek<'a>(&'a RefCell<Option<String>>);
    impl Decider for Peek<'_> {
        fn decide(&self, prompt: &Prompt, _pv: PolicyVersion) -> Result<Completion, AgentError> {
            *self.0.borrow_mut() = Some(format!("{}{}", prompt.system(), prompt.user()));
            Err(AgentError::Transport("stop after peeking".into()))
        }
    }
    let _ = run(&mut r, &repo, &task, &Peek(&seen));

    let text = seen.borrow().clone().expect("the agent was called");
    assert!(
        !text.contains(&task.commit),
        "the solving commit sha reached the prompt"
    );
}

#[test]
fn observation_authority_is_what_the_run_declares() {
    // A sealed solo episode reads back its own work, so Agent is correct.
    // Defaulting to Observed would make every step after the first need a
    // human and unattended runs impossible.
    let (repo, task) = task_fixture();
    let mut r = rig();
    let agent = Scripted::new(vec![noop_draft(0.5); 2]);
    let _ = run(&mut r, &repo, &task, &agent);
    assert_eq!(*agent.authority_seen.borrow().first().unwrap(), Authority::Agent);

    let mut r2 = rig();
    r2.cfg.observation_authority = Authority::Observed;
    let agent2 = Scripted::new(vec![noop_draft(0.5); 2]);
    let _ = run(&mut r2, &repo, &task, &agent2);
    assert_eq!(*agent2.authority_seen.borrow().first().unwrap(), Authority::Observed);
}

#[test]
fn the_whole_episode_is_in_the_ledger_and_verifies() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    let agent = Scripted::new(vec![fix_draft(0.9)]);
    let _ = run(&mut r, &repo, &task, &agent);

    assert_eq!(r.ledger.by_kind("decision_recorded").unwrap().len(), 1);
    assert_eq!(r.ledger.by_kind("outcome_observed").unwrap().len(), 1);
    assert_eq!(r.ledger.by_kind("episode_scored").unwrap().len(), 1);
    r.ledger.verify().unwrap();

    // Calibration is computable from the ledger alone, without the episode
    // having to hand it over.
    let pairs = r.ledger.calibration_pairs().unwrap();
    assert_eq!(pairs, vec![(0.9, true)]);
}

#[test]
fn the_sandbox_is_cleaned_up() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    let root = r.cfg.scratch_root.clone();
    let agent = Scripted::new(vec![fix_draft(0.9)]);
    let _ = run(&mut r, &repo, &task, &agent);

    let left: Vec<PathBuf> = std::fs::read_dir(&root)
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "episode left directories behind: {left:?}");
}

#[test]
fn tokens_are_charged_against_the_budget() {
    let (repo, task) = task_fixture();
    let mut r = rig();
    let before = r.router.budget().remaining();
    let agent = Scripted::new(vec![noop_draft(0.5); 3]);
    let out = run(&mut r, &repo, &task, &agent);

    assert_eq!(out.tokens, 3 * 300, "100 prompt + 200 completion per call");
    assert_eq!(r.router.budget().remaining(), before - out.tokens);
}
