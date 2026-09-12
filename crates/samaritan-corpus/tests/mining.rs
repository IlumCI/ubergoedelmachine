//! Tests for corpus mining, against a real git repository built per test.
//!
//! The sandbox tests are the ones that matter. A leaked solution does not
//! crash anything — it produces an agent that appears to have improved
//! dramatically — so the seal is proven here rather than assumed, including
//! one test that deliberately breaks it to confirm the check would notice.

use std::path::{Path, PathBuf};
use std::process::Command;

use samaritan_corpus::screen::{OracleOutcome, OracleRunner, Screening, screen_one};
use samaritan_corpus::{
    AnswerKind, Bucket, Corpus, Manifest, MineOptions, OracleSpec, Split, Task, TaskId, TaskKind,
    TestPatterns, grade_answer, materialize, mine, screen_for_flakes,
};

// ------------------------------------------------------------- test repo

struct Repo {
    dir: tempfile::TempDir,
}

impl Repo {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let r = Repo { dir };
        r.git(["init", "-q", "-b", "main", "."]);
        r.git(["config", "user.email", "t@example.invalid"]);
        r.git(["config", "user.name", "Test"]);
        r.git(["config", "commit.gpgsign", "false"]);
        r
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn git<I: IntoIterator<Item = S>, S: AsRef<std::ffi::OsStr>>(&self, args: I) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(self.path())
            .args(args)
            .output()
            .expect("git must be on PATH");
        assert!(
            out.status.success(),
            "git failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn write(&self, rel: &str, content: &str) {
        let p = self.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    fn rm(&self, rel: &str) {
        std::fs::remove_file(self.path().join(rel)).unwrap();
    }

    /// Commit with an explicit date so time-based splitting is deterministic.
    fn commit(&self, message: &str, date: &str) -> String {
        self.git(["add", "-A"]);
        let out = Command::new("git")
            .arg("-C")
            .arg(self.path())
            .args(["commit", "-q", "--allow-empty", "-m", message])
            .env("GIT_AUTHOR_DATE", date)
            .env("GIT_COMMITTER_DATE", date)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "commit failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        self.git(["rev-parse", "HEAD"])
    }
}

/// A repo with one clean task in it: a source+test commit preceded by a base.
fn repo_with_one_task() -> (Repo, String) {
    let r = Repo::new();
    r.write("src/lib.rs", "pub fn add(a: i32, b: i32) -> i32 { a - b }\n");
    r.write("tests/add.rs", "#[test] fn t() { assert_eq!(1, 1); }\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.write("src/lib.rs", "pub fn add(a: i32, b: i32) -> i32 { a + b }\n");
    r.write(
        "tests/add.rs",
        "#[test] fn t() { assert_eq!(samaritan::add(2,2), 4); }\n",
    );
    let solve = r.commit("fix addition sign error", "2026-02-01T00:00:00+00:00");
    (r, solve)
}

fn opts() -> MineOptions {
    MineOptions::new("fixture", "2026-06-01T00:00:00+00:00")
}

// -------------------------------------------------------- classification

#[test]
fn test_paths_are_recognised_across_conventions() {
    let p = TestPatterns::default();
    for path in [
        "tests/foo.rs",
        "src/__tests__/x.js",
        "pkg/test/helper.go",
        "test_parser.py",
        "src/parser_test.go",
        "app/thing.test.ts",
        "app/thing.spec.js",
    ] {
        assert!(p.is_test(path), "{path} should be a test");
    }
}

#[test]
fn source_paths_are_not_mistaken_for_tests() {
    let p = TestPatterns::default();
    for path in [
        "src/lib.rs",
        "src/latest.rs",     // ends in "test" but is not a test
        "contest/main.go",   // contains "test" as a substring of a segment
        "src/protester.py",
    ] {
        assert!(!p.is_test(path), "{path} should not be a test");
    }
}

#[test]
fn windows_separators_classify_the_same_way() {
    let p = TestPatterns::default();
    assert!(p.is_test(r"tests\add.rs"));
    assert!(!p.is_test(r"src\lib.rs"));
}

// --------------------------------------------------------------- mining

#[test]
fn a_source_plus_test_commit_becomes_a_task() {
    let (r, solve) = repo_with_one_task();
    let c = mine(r.path(), &opts()).unwrap();

    assert_eq!(c.tasks.len(), 1, "exactly one commit qualifies");
    let t = &c.tasks[0];
    assert_eq!(t.commit, solve);
    assert_eq!(t.prompt, "fix addition sign error");
    assert_eq!(t.test_paths, vec!["tests/add.rs"]);
    assert_eq!(t.source_paths, vec!["src/lib.rs"]);
    assert_eq!(t.split, Split::Train);
}

#[test]
fn commits_without_both_halves_are_skipped() {
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() {}\n");
    r.write("tests/t.rs", "#[test] fn t() {}\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.write("src/lib.rs", "fn a() { /* refactor */ }\n");
    r.commit("source only, no oracle to settle it", "2026-01-02T00:00:00+00:00");

    r.write("tests/t.rs", "#[test] fn t() { assert!(true); }\n");
    r.commit("tests only, nothing to fix", "2026-01-03T00:00:00+00:00");

    assert!(
        mine(r.path(), &opts()).unwrap().tasks.is_empty(),
        "neither commit is a task"
    );
}

#[test]
fn the_root_commit_is_never_a_task() {
    // It has no parent, so there is no state to start the agent from.
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() {}\n");
    r.write("tests/t.rs", "#[test] fn t() {}\n");
    r.commit("initial with both", "2026-01-01T00:00:00+00:00");
    assert!(mine(r.path(), &opts()).unwrap().tasks.is_empty());
}

#[test]
fn the_prompt_is_the_subject_and_the_body_is_withheld() {
    // Commit bodies routinely describe the fix. Including one would turn the
    // task into reading comprehension.
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() -> i32 { 0 }\n");
    r.write("tests/t.rs", "#[test] fn t() {}\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.write("src/lib.rs", "fn a() -> i32 { 42 }\n");
    r.write("tests/t.rs", "#[test] fn t() { assert_eq!(a(), 42); }\n");
    r.commit(
        "fix the return value\n\nThe bug was that a() returned 0. Change the literal to 42.",
        "2026-01-02T00:00:00+00:00",
    );

    let c = mine(r.path(), &opts()).unwrap();
    let prompt = &c.tasks[0].prompt;
    assert_eq!(prompt, "fix the return value");
    assert!(!prompt.contains("42"), "the body leaked the answer: {prompt}");
    assert!(!prompt.contains("literal"));
}

#[test]
fn the_split_is_taken_on_commit_time() {
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() {}\n");
    r.write("tests/t.rs", "#[test] fn t() {}\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.write("src/lib.rs", "fn a() { 1; }\n");
    r.write("tests/t.rs", "#[test] fn t() { assert!(true) }\n");
    r.commit("early fix", "2026-03-01T00:00:00+00:00");

    r.write("src/lib.rs", "fn a() { 2; }\n");
    r.write("tests/t.rs", "#[test] fn t() { assert!(1 == 1) }\n");
    r.commit("late fix", "2026-09-01T00:00:00+00:00");

    let c = mine(r.path(), &opts()).unwrap();
    let train: Vec<_> = c.tasks_in(Split::Train).map(|t| &t.prompt).collect();
    let held: Vec<_> = c.tasks_in(Split::HeldOut).map(|t| &t.prompt).collect();

    assert_eq!(train, vec!["early fix"]);
    assert_eq!(held, vec!["late fix"], "later commits are the yardstick");
}

#[test]
fn deleted_test_files_are_tracked_separately() {
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() {}\n");
    r.write("tests/old.rs", "#[test] fn old() {}\n");
    r.write("tests/keep.rs", "#[test] fn keep() {}\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.rm("tests/old.rs");
    r.write("src/lib.rs", "fn a() { 1; }\n");
    r.write("tests/keep.rs", "#[test] fn keep() { assert!(true) }\n");
    r.commit("drop the obsolete test", "2026-01-02T00:00:00+00:00");

    let c = mine(r.path(), &opts()).unwrap();
    let t = &c.tasks[0];
    assert_eq!(t.deleted_test_paths, vec!["tests/old.rs"]);
    assert_eq!(t.test_paths, vec!["tests/keep.rs"]);
}

#[test]
fn difficulty_counts_source_churn_and_ignores_test_churn() {
    // The agent is handed the tests, so a huge fixture is not work it does.
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() {}\n");
    r.write("tests/t.rs", "#[test] fn t() {}\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.write("src/lib.rs", "fn a() { 1; }\n");
    let big: String = (0..200).map(|i| format!("// fixture line {i}\n")).collect();
    r.write("tests/t.rs", &big);
    r.commit("small fix, huge fixture", "2026-01-02T00:00:00+00:00");

    let c = mine(r.path(), &opts()).unwrap();
    let d = c.tasks[0].difficulty;
    assert_eq!(d.source_files_changed, 1);
    assert!(
        d.lines_added < 10,
        "200 lines of test fixture must not read as difficulty: {d:?}"
    );
    assert_eq!(d.bucket(), Bucket::Trivial);
}

#[test]
fn the_limit_is_respected() {
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() {}\n");
    r.write("tests/t.rs", "#[test] fn t() {}\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");
    for i in 0..5 {
        r.write("src/lib.rs", &format!("fn a() {{ {i}; }}\n"));
        r.write("tests/t.rs", &format!("#[test] fn t() {{ let _ = {i}; }}\n"));
        r.commit(&format!("fix {i}"), "2026-01-02T00:00:00+00:00");
    }
    let mut o = opts();
    o.limit = Some(2);
    assert_eq!(mine(r.path(), &o).unwrap().tasks.len(), 2);
}

// -------------------------------------------------------------- sandbox

fn materialize_one(r: &Repo, t: &Task) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("work");
    let sb = materialize(r.path(), t, &root).unwrap();
    sb.assert_sealed().unwrap();
    (dir, root)
}

#[test]
fn the_agent_gets_new_tests_over_old_source() {
    let (r, _) = repo_with_one_task();
    let c = mine(r.path(), &opts()).unwrap();
    let (_guard, root) = materialize_one(&r, &c.tasks[0]);

    let src = std::fs::read_to_string(root.join("src/lib.rs")).unwrap();
    let test = std::fs::read_to_string(root.join("tests/add.rs")).unwrap();

    assert!(src.contains("a - b"), "source must be the unfixed parent");
    assert!(
        test.contains("add(2,2), 4"),
        "tests must be the commit's new ones"
    );
}

#[test]
fn the_sandbox_cannot_see_the_solution() {
    // The central guarantee. A `git worktree` would fail this test, which is
    // exactly why the sandbox is not one.
    let (r, solve) = repo_with_one_task();
    let c = mine(r.path(), &opts()).unwrap();
    let (_guard, root) = materialize_one(&r, &c.tasks[0]);

    let out = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["cat-file", "-e", &format!("{solve}^{{commit}}")])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "the solving commit is reachable from the sandbox"
    );
}

#[test]
fn the_sandbox_has_no_route_to_fetch_more() {
    let (r, _) = repo_with_one_task();
    let c = mine(r.path(), &opts()).unwrap();
    let (_guard, root) = materialize_one(&r, &c.tasks[0]);

    let remotes = Command::new("git")
        .arg("-C")
        .arg(&root)
        .arg("remote")
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&remotes.stdout).trim().is_empty(),
        "a remote would let the agent fetch the answer"
    );
}

#[test]
fn the_sandbox_history_is_shallow() {
    let (r, _) = repo_with_one_task();
    let c = mine(r.path(), &opts()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("work");
    let sb = materialize(r.path(), &c.tasks[0], &root).unwrap();

    // The fetched parent, plus the commit that applies the task's tests.
    assert!(
        sb.visible_history().unwrap() <= 2,
        "the agent should see almost no history"
    );
}

#[test]
fn a_broken_seal_is_detected_rather_than_trusted() {
    // Deliberately leak the solution into a materialised sandbox and confirm
    // the runtime check catches it. Without this, `assert_sealed` could be
    // vacuously true and every other sandbox test would still pass.
    let (r, solve) = repo_with_one_task();
    let c = mine(r.path(), &opts()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("work");
    let sb = materialize(r.path(), &c.tasks[0], &root).unwrap();
    sb.assert_sealed().expect("sealed to begin with");

    let out = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args([
            "-c",
            "uploadpack.allowAnySHA1InWant=true",
            "-c",
            "protocol.file.allow=always",
            "fetch",
            "-q",
            "--depth",
            "1",
            &r.path().to_string_lossy(),
            &solve,
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "the leak itself should succeed");

    assert!(
        sb.assert_sealed().is_err(),
        "assert_sealed must notice the solution arriving"
    );
}

#[test]
fn deleted_tests_are_removed_from_the_sandbox() {
    let r = Repo::new();
    r.write("src/lib.rs", "fn a() {}\n");
    r.write("tests/old.rs", "#[test] fn old() { panic!() }\n");
    r.write("tests/keep.rs", "#[test] fn keep() {}\n");
    r.commit("initial", "2026-01-01T00:00:00+00:00");

    r.rm("tests/old.rs");
    r.write("src/lib.rs", "fn a() { 1; }\n");
    r.write("tests/keep.rs", "#[test] fn keep() { assert!(true) }\n");
    r.commit("drop the obsolete test", "2026-01-02T00:00:00+00:00");

    let c = mine(r.path(), &opts()).unwrap();
    let (_guard, root) = materialize_one(&r, &c.tasks[0]);

    assert!(
        !root.join("tests/old.rs").exists(),
        "a test the commit deleted must not fail the sandbox suite"
    );
    assert!(root.join("tests/keep.rs").exists());
}

// ------------------------------------------------------------ screening

/// A runner following a script, so screening logic is testable without
/// building anything.
struct Scripted {
    at_solution: Vec<OracleOutcome>,
    before_fix: OracleOutcome,
    solution_calls: usize,
}

impl Scripted {
    fn new(at_solution: Vec<OracleOutcome>, before_fix: OracleOutcome) -> Self {
        Self {
            at_solution,
            before_fix,
            solution_calls: 0,
        }
    }
}

impl OracleRunner for Scripted {
    fn run(&mut self, _task: &Task, at_solution: bool) -> OracleOutcome {
        if at_solution {
            let i = self.solution_calls.min(self.at_solution.len() - 1);
            self.solution_calls += 1;
            self.at_solution[i]
        } else {
            self.before_fix
        }
    }
}

fn a_task() -> Task {
    let (r, _) = repo_with_one_task();
    mine(r.path(), &opts()).unwrap().tasks.remove(0)
}

#[test]
fn a_well_formed_task_is_admitted() {
    let t = a_task();
    let mut runner = Scripted::new(
        vec![OracleOutcome::Passed, OracleOutcome::Passed],
        OracleOutcome::Failed,
    );
    assert_eq!(screen_one(&mut runner, &t), Screening::Admitted);
}

#[test]
fn a_flaky_oracle_is_rejected() {
    // Two runs of the same state disagreeing means every utility estimate
    // this task contributes to would carry noise the certificate cannot see.
    let t = a_task();
    let mut runner = Scripted::new(
        vec![OracleOutcome::Passed, OracleOutcome::Failed],
        OracleOutcome::Failed,
    );
    assert!(matches!(
        screen_one(&mut runner, &t),
        Screening::Flaky { .. }
    ));
}

#[test]
fn a_task_that_already_passes_is_rejected() {
    let t = a_task();
    let mut runner = Scripted::new(
        vec![OracleOutcome::Passed, OracleOutcome::Passed],
        OracleOutcome::Passed,
    );
    assert_eq!(screen_one(&mut runner, &t), Screening::NoFailingOracle);
}

#[test]
fn a_solution_that_does_not_pass_is_rejected() {
    let t = a_task();
    let mut runner = Scripted::new(vec![OracleOutcome::Failed], OracleOutcome::Failed);
    assert!(matches!(
        screen_one(&mut runner, &t),
        Screening::SolutionDoesNotPass { .. }
    ));
}

#[test]
fn an_unbuildable_task_is_distinguished_from_a_failing_one() {
    let t = a_task();
    let mut runner = Scripted::new(vec![OracleOutcome::Errored], OracleOutcome::Failed);
    assert!(matches!(
        screen_one(&mut runner, &t),
        Screening::Unrunnable { .. }
    ));
}

#[test]
fn screening_removes_rejected_tasks_from_the_corpus() {
    let (r, _) = repo_with_one_task();
    let mut c = mine(r.path(), &opts()).unwrap();
    assert_eq!(c.tasks.len(), 1);

    let mut runner = Scripted::new(
        vec![OracleOutcome::Passed, OracleOutcome::Failed],
        OracleOutcome::Failed,
    );
    let report = screen_for_flakes(&mut runner, &mut c);

    assert!(c.tasks.is_empty(), "the flaky task must be gone");
    assert_eq!(report.admission_rate(), 0.0);
    assert_eq!(report.rejection_breakdown().get("flaky"), Some(&1));
}

// ------------------------------------------------------------- manifest

fn corpus(name: &str, repo: &str, trainable: bool) -> Corpus {
    Corpus {
        name: name.into(),
        repo_path: repo.into(),
        cutoff: "2026-06-01T00:00:00+00:00".into(),
        trainable,
        tasks: vec![],
    }
}

#[test]
fn training_and_transfer_corpora_may_not_share_a_repository() {
    // The most damaging mistake available, and the easiest to make by
    // accident: transfer numbers from a repository the agent trained on are
    // not transfer numbers, and nothing downstream can tell.
    let m = Manifest {
        corpora: vec![
            corpus("seen", "/repos/ripgrep", true),
            corpus("unseen", "/repos/ripgrep", false),
        ],
    };
    let err = m.check_disjoint().unwrap_err();
    assert!(err.to_string().contains("both training and transfer"));

    let ok = Manifest {
        corpora: vec![
            corpus("seen", "/repos/ripgrep", true),
            corpus("unseen", "/repos/fd", false),
        ],
    };
    ok.check_disjoint().unwrap();
    assert_eq!(ok.transfer_corpora().count(), 1);
}

#[test]
fn a_transfer_corpus_yields_no_trainable_tasks_even_in_the_train_split() {
    // Defence in depth: a caller that forgets to check `trainable` gets
    // nothing rather than silently contaminating the experiment.
    let (r, _) = repo_with_one_task();
    let mut o = opts().transfer_only();
    o.corpus_name = "unseen".into();
    let c = mine(r.path(), &o).unwrap();

    assert_eq!(c.tasks_in(Split::Train).count(), 1, "the task exists");
    assert!(
        c.trainable_tasks().is_empty(),
        "but it is not available for training"
    );
}

#[test]
fn the_manifest_digest_changes_with_its_contents() {
    let a = Manifest {
        corpora: vec![corpus("x", "/r", true)],
    };
    let b = Manifest {
        corpora: vec![corpus("x", "/r", false)],
    };
    assert_ne!(
        a.digest(),
        b.digest(),
        "a run that changed its corpora must not claim the same manifest"
    );
}

#[test]
fn tasks_can_be_drawn_stratified_by_difficulty() {
    let (r, _) = repo_with_one_task();
    let c = mine(r.path(), &opts()).unwrap();
    let strata = c.stratified(Split::Train);
    assert_eq!(strata.values().map(|v| v.len()).sum::<usize>(), 1);
    assert!(strata.contains_key(&Bucket::Trivial));
}

#[test]
fn oracle_specs_carry_a_timeout() {
    // A hung suite is a failed episode, not a reason to block the run.
    assert!(OracleSpec::cargo_test().timeout_secs > 0);
    assert!(OracleSpec::pytest().timeout_secs > 0);
}

// ------------------------------------------------------------- frontier

use samaritan_corpus::{SolvabilityWitness, WitnessKind};

fn frontier_task(witness: Option<SolvabilityWitness>) -> Task {
    let (r, _) = repo_with_one_task();
    let mut t = mine(r.path(), &opts()).unwrap().tasks.remove(0);
    t.split = Split::Frontier;
    t.witness = witness;
    t
}

#[test]
fn a_frontier_task_without_a_witness_is_rejected() {
    // An unsolved unwitnessed task produces exactly the same data as an
    // unsolved witnessed one, and only the second is evidence of anything.
    let mut c = corpus("frontier", "/repos/x", false);
    c.tasks.push(frontier_task(None));
    let err = c.check_frontier_witnessed().unwrap_err();
    assert!(err.to_string().contains("solvability witness"));
}

#[test]
fn a_witnessed_frontier_task_is_accepted() {
    let mut c = corpus("frontier", "/repos/x", false);
    c.tasks.push(frontier_task(Some(SolvabilityWitness {
        kind: WitnessKind::ReachableScaffold {
            mutation_sketch: "add a retrieval lesson that chunks the file before editing".into(),
        },
        evidence: "solved once at temperature 0 with the chunking lesson pre-installed".into(),
        recorded_at: "2026-09-01T00:00:00+00:00".into(),
    })));
    c.check_frontier_witnessed().unwrap();
}

#[test]
fn frontier_tasks_never_reach_the_training_loop() {
    // Two independent guards: the corpus is marked untrainable, and
    // trainable_tasks only ever returns the Train split.
    let mut trainable = corpus("mixed", "/repos/x", true);
    trainable.tasks.push(frontier_task(None));
    assert!(
        trainable.trainable_tasks().is_empty(),
        "a frontier task must not be trainable even in a trainable corpus"
    );
    assert_eq!(trainable.tasks_in(Split::Frontier).count(), 1);
}

// ------------------------------------------------------- the reasoning surface

fn reasoning_task(answer: &str, kind: AnswerKind) -> Task {
    Task::reasoning(
        TaskId("r1".into()),
        "numina",
        "What is 6 times 7?",
        answer,
        kind,
        "math",
        "2026-09-01T00:00:00+00:00",
        Split::Train,
    )
}

#[test]
fn a_reasoning_task_grades_its_answer_and_a_coding_task_does_not() {
    let r = reasoning_task("42", AnswerKind::ExactMatch);
    assert!(r.is_reasoning());
    assert_eq!(r.domain(), "math");
    assert_eq!(r.grade("The answer is 42."), Some(true));
    assert_eq!(r.grade("43"), Some(false));

    // A coding task has no answer key; grading one is a category error and
    // returns None rather than a misleading verdict.
    let (rr, _) = repo_with_one_task();
    let coding = mine(rr.path(), &opts()).unwrap().tasks.remove(0);
    assert!(!coding.is_reasoning());
    assert_eq!(coding.grade("anything"), None);
}

#[test]
fn grade_answer_is_conservative() {
    // A single-token key is found as a standalone token; a multi-word key needs
    // the whole phrase; a coincidental overlap does not count.
    assert!(grade_answer("C", AnswerKind::MultipleChoice, "the correct option is C) foo"));
    assert!(!grade_answer("C", AnswerKind::MultipleChoice, "I pick B"));
    assert!(grade_answer("Marie Curie", AnswerKind::ExactMatch, "it was marie curie"));
    assert!(!grade_answer("Marie Curie", AnswerKind::ExactMatch, "curie units are unrelated"));
    assert!(!grade_answer("", AnswerKind::ExactMatch, "anything"));
}

#[test]
fn grade_answer_compares_numbers_numerically() {
    // The exact miss the first p2 eval exposed: "45" answers the key "45.0".
    assert!(grade_answer("45.0", AnswerKind::ExactMatch, "45"));
    assert!(grade_answer("45.0", AnswerKind::ExactMatch, "The answer is 45."));
    // Units and LaTeX around the number don't matter; the number does.
    assert!(grade_answer("35", AnswerKind::ExactMatch, r"\boxed{35\%}"));
    assert!(grade_answer("20", AnswerKind::ExactMatch, "20%"));
    assert!(grade_answer("1024", AnswerKind::ExactMatch, "1,024"));
    // A genuinely different number is still wrong.
    assert!(!grade_answer("37.5", AnswerKind::ExactMatch, "66%"));
    assert!(!grade_answer("42.5", AnswerKind::ExactMatch, "65%"));
    // A numeric key is judged only on the number — a text key still isn't:
    // "yes"/"C" keep the token match, unaffected by the numeric path.
    assert!(grade_answer("yes", AnswerKind::ExactMatch, "Yes, all of them."));
}

#[test]
fn a_task_without_a_kind_field_deserializes_as_coding() {
    // Backward compatibility: every task mined or stored before the reasoning
    // surface existed carried no `kind`, and must still read as a coding task.
    let json = serde_json::json!({
        "id": "legacy@abc",
        "corpus": "ripgrep",
        "commit": "abc",
        "parent": "def",
        "prompt": "fix the bug",
        "test_paths": ["tests/x.rs"],
        "deleted_test_paths": [],
        "source_paths": ["src/x.rs"],
        "oracle": { "program": "cargo", "args": ["test"], "timeout_secs": 300 },
        "difficulty": { "source_files_changed": 1, "lines_added": 3, "lines_removed": 1, "test_files_changed": 1 },
        "committed_at": "2026-01-01T00:00:00+00:00",
        "split": "train"
    });
    let t: Task = serde_json::from_value(json).expect("legacy task deserializes");
    assert!(matches!(t.kind, TaskKind::Coding));
    assert!(!t.is_reasoning());
}
