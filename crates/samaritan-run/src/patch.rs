//! The effectful half of level 3: trying a patch somewhere that is not here.
//!
//! [`samaritan_search::gate_code_patch`] decides *whether* a patch may land;
//! this is the part that finds out whether it builds and passes, and the whole
//! contract is that it does so **without touching the tree the harness is
//! running from**.
//!
//! The isolation is a linked git worktree checked out at `HEAD`. That is not a
//! copy of the working directory: it is a clean tree at the committed state, so
//! a patch is judged against what is actually committed rather than against
//! whatever happens to be uncommitted at the time. A patch that only applies to
//! a dirty tree is a patch that would not survive a fresh checkout, and finding
//! that out here is the point.
//!
//! The build and test commands are configurable, which is not a generality for
//! its own sake: it is what lets the worktree plumbing be tested in seconds
//! against a trivial command instead of minutes against a full `cargo test`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use samaritan_exec::proc;
use samaritan_search::{PatchReport, PatchVerifier};

/// Verifies a candidate patch in a throwaway git worktree.
pub struct WorktreeVerifier {
    repo: PathBuf,
    scratch: PathBuf,
    build: Vec<String>,
    test: Vec<String>,
    timeout: Duration,
}

impl WorktreeVerifier {
    /// `repo` is the repository to branch a worktree from; `scratch` is where
    /// that worktree is created and must not already exist.
    pub fn new(repo: impl Into<PathBuf>, scratch: impl Into<PathBuf>) -> Self {
        Self {
            repo: repo.into(),
            scratch: scratch.into(),
            build: ["cargo", "build", "--workspace", "--lib"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            test: ["cargo", "test", "--workspace", "--lib", "--tests"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            // A machine-authored patch can produce a build that never
            // terminates; the timeout is the thing that makes this safe to run
            // unattended.
            timeout: Duration::from_secs(1800),
        }
    }

    /// Override the build command. First element is the program.
    pub fn with_build(mut self, cmd: Vec<String>) -> Self {
        self.build = cmd;
        self
    }

    /// Override the test command.
    pub fn with_test(mut self, cmd: Vec<String>) -> Self {
        self.test = cmd;
        self
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    fn git(&self, args: &[&str], cwd: &Path) -> proc::Output {
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        proc::run("git", &owned, cwd, Duration::from_secs(120), &[])
    }

    fn run_cmd(&self, cmd: &[String], cwd: &Path) -> Option<proc::Output> {
        let (program, args) = cmd.split_first()?;
        Some(proc::run(program, args, cwd, self.timeout, &[]))
    }

    /// Remove the worktree, whatever happened. Best-effort: a leaked scratch
    /// tree is untidy, but failing to clean up must not turn a refusal into an
    /// acceptance or vice versa.
    fn cleanup(&self) {
        let scratch = self.scratch.to_string_lossy().to_string();
        let _ = self.git(&["worktree", "remove", "--force", &scratch], &self.repo);
        if self.scratch.exists() {
            let _ = std::fs::remove_dir_all(&self.scratch);
        }
        let _ = self.git(&["worktree", "prune"], &self.repo);
    }
}

impl PatchVerifier for WorktreeVerifier {
    fn verify(&mut self, unified_diff: &str) -> PatchReport {
        // Start clean: a leftover worktree from a previous run would otherwise
        // be judged instead of this patch.
        self.cleanup();

        let scratch = self.scratch.to_string_lossy().to_string();
        let add = self.git(&["worktree", "add", "--detach", &scratch, "HEAD"], &self.repo);
        if !add.success() {
            self.cleanup();
            return PatchReport::failed_to_build(format!(
                "could not create a scratch worktree: {}",
                add.stderr.trim()
            ));
        }

        // Apply the diff *inside* the worktree. `git apply` refuses anything it
        // cannot apply cleanly, which is the answer we want: a patch that does
        // not apply has failed, and guessing at it would be worse than failing.
        let patch_file = self.scratch.join(".samaritan-candidate.patch");
        if let Err(e) = std::fs::write(&patch_file, unified_diff) {
            self.cleanup();
            return PatchReport::failed_to_build(format!("could not stage the patch: {e}"));
        }
        let applied = self.git(
            &["apply", "--whitespace=nowarn", ".samaritan-candidate.patch"],
            &self.scratch,
        );
        let _ = std::fs::remove_file(&patch_file);
        if !applied.success() {
            let detail = format!("the patch did not apply: {}", applied.stderr.trim());
            self.cleanup();
            return PatchReport::failed_to_build(detail);
        }

        let Some(built) = self.run_cmd(&self.build.clone(), &self.scratch) else {
            self.cleanup();
            return PatchReport::failed_to_build("no build command configured");
        };
        if !built.success() {
            let detail = tail(&built.stderr, &built.stdout);
            self.cleanup();
            return PatchReport::failed_to_build(detail);
        }

        let Some(tested) = self.run_cmd(&self.test.clone(), &self.scratch) else {
            self.cleanup();
            return PatchReport::failed_to_build("no test command configured");
        };
        let report = if tested.success() {
            PatchReport::passed(tail(&tested.stdout, &tested.stderr))
        } else {
            PatchReport::tests_failed(tail(&tested.stderr, &tested.stdout))
        };
        self.cleanup();
        report
    }
}

/// The last few lines of output — enough for a human to judge, short enough to
/// sit in a ledger row.
fn tail(primary: &str, fallback: &str) -> String {
    let src = if primary.trim().is_empty() { fallback } else { primary };
    let lines: Vec<&str> = src.lines().collect();
    let start = lines.len().saturating_sub(20);
    lines[start..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use samaritan_dsl::Mutation;
    use samaritan_kernel::{Admission, Approval};
    use samaritan_search::gate_code_patch;

    /// A tiny git repo with one file, so the worktree machinery can be
    /// exercised in milliseconds rather than against the real workspace.
    fn scratch_repo() -> Option<(tempfile::TempDir, PathBuf)> {
        let dir = tempfile::tempdir().ok()?;
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).ok()?;
        let run = |args: &[&str]| {
            let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
            proc::run("git", &owned, &repo, Duration::from_secs(60), &[])
        };
        if !run(&["init", "-q"]).success() {
            return None; // no git available; the test skips itself
        }
        let _ = run(&["config", "user.email", "t@example.com"]);
        let _ = run(&["config", "user.name", "t"]);
        std::fs::write(repo.join("hello.txt"), "one\n").ok()?;
        let _ = run(&["add", "."]);
        if !run(&["commit", "-qm", "init"]).success() {
            return None;
        }
        Some((dir, repo))
    }

    /// A diff that applies cleanly to the scratch repo above.
    const GOOD_DIFF: &str = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n-one\n+two\n";

    fn verifier(repo: &Path, scratch: PathBuf) -> WorktreeVerifier {
        // Trivial commands: this test is about the worktree plumbing, not about
        // whether cargo works.
        WorktreeVerifier::new(repo, scratch)
            .with_build(vec!["git".into(), "--version".into()])
            .with_test(vec!["git".into(), "--version".into()])
            .with_timeout(Duration::from_secs(60))
    }

    #[test]
    fn a_patch_that_applies_and_passes_is_reported_green() {
        let Some((dir, repo)) = scratch_repo() else { return };
        let mut v = verifier(&repo, dir.path().join("wt1"));
        let r = v.verify(GOOD_DIFF);
        assert!(r.builds && r.tests_pass, "{r:?}");
    }

    #[test]
    fn a_patch_that_does_not_apply_fails_rather_than_being_guessed_at() {
        let Some((dir, repo)) = scratch_repo() else { return };
        let mut v = verifier(&repo, dir.path().join("wt2"));
        let r = v.verify("--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n-NOT THE CONTENT\n+x\n");
        assert!(!r.builds, "{r:?}");
        assert!(r.detail.contains("did not apply"), "{}", r.detail);
    }

    #[test]
    fn the_live_tree_is_never_modified() {
        // The contract. After a verify, the file in the real repo still reads
        // what it did before — the change happened only in the worktree.
        let Some((dir, repo)) = scratch_repo() else { return };
        let mut v = verifier(&repo, dir.path().join("wt3"));
        let _ = v.verify(GOOD_DIFF);
        let live = std::fs::read_to_string(repo.join("hello.txt")).unwrap();
        assert_eq!(live, "one\n", "the patch must not touch the live tree");
    }

    #[test]
    fn the_scratch_worktree_is_cleaned_up() {
        let Some((dir, repo)) = scratch_repo() else { return };
        let scratch = dir.path().join("wt4");
        let mut v = verifier(&repo, scratch.clone());
        let _ = v.verify(GOOD_DIFF);
        assert!(!scratch.exists(), "the scratch worktree should be removed");
    }

    #[test]
    fn a_failing_test_command_reports_tests_failed_not_build_failed() {
        let Some((dir, repo)) = scratch_repo() else { return };
        let mut v = WorktreeVerifier::new(&repo, dir.path().join("wt5"))
            .with_build(vec!["git".into(), "--version".into()])
            .with_test(vec!["git".into(), "--no-such-subcommand".into()])
            .with_timeout(Duration::from_secs(60));
        let r = v.verify(GOOD_DIFF);
        assert!(r.builds, "the build step succeeded");
        assert!(!r.tests_pass, "the test step must be reported as failing");
    }

    #[test]
    fn end_to_end_the_gate_still_demands_a_human() {
        // The real verifier wired to the real gate: green, and still refused
        // without approval.
        let Some((dir, repo)) = scratch_repo() else { return };
        let mut v = verifier(&repo, dir.path().join("wt6"));
        let m = Mutation::CodePatch { unified_diff: GOOD_DIFF.into() };
        assert!(gate_code_patch(&m, &Admission::new(3), &mut v, Approval::Refuse).is_err());
        assert!(gate_code_patch(&m, &Admission::new(3), &mut v, Approval::Allow).is_ok());
    }
}
