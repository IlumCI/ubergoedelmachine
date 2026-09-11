//! Materialising a task into an environment that does not contain its answer.
//!
//! The obvious implementation — `git worktree add` at the parent commit — is
//! wrong, and quietly so. A worktree shares the object database of the
//! repository it came from, so the solving commit is still there: `git log
//! --all`, `git show <sha>`, or simply `git diff HEAD origin/main` hands the
//! agent the diff it was asked to reproduce. The episode still runs, the tests
//! still pass, and the resulting number means nothing.
//!
//! So the sandbox is built the other way around. A fresh repository fetches
//! *only* the parent commit, shallowly, and the task's test files are written
//! in as plain bytes read out of the source repository by the miner. No object
//! belonging to the solving commit is ever transferred. The agent's environment
//! contains exactly one commit, no remote, and no route to the answer.
//!
//! [`Sandbox::assert_sealed`] re-checks that at runtime rather than trusting
//! the construction, because this is the property whose failure is least
//! visible and most expensive.

use std::fs;
use std::path::{Path, PathBuf};

use crate::git::{git, git_bytes, has_commit};
use crate::{CorpusError, Task};

/// A prepared task environment.
pub struct Sandbox {
    pub root: PathBuf,
    pub task_id: crate::TaskId,
    /// Kept so the seal can be re-verified at any point in the episode.
    solving_commit: String,
}

impl Sandbox {
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Verify the answer is genuinely absent.
    ///
    /// Cheap, and called after construction on every episode. The failure this
    /// catches — a leaked solution — does not announce itself: it shows up as
    /// an agent that appears to have gotten much better.
    pub fn assert_sealed(&self) -> Result<(), CorpusError> {
        if has_commit(&self.root, &self.solving_commit) {
            return Err(CorpusError::Invalid(format!(
                "sandbox for {} can see its own solution ({}); the episode would be meaningless",
                self.task_id, self.solving_commit
            )));
        }

        let remotes = git(&self.root, ["remote"])?;
        if !remotes.trim().is_empty() {
            return Err(CorpusError::Invalid(format!(
                "sandbox for {} has remotes configured ({}); it could fetch the solution",
                self.task_id,
                remotes.replace('\n', ", ")
            )));
        }

        Ok(())
    }

    /// How many commits the agent can see. Should be exactly one.
    pub fn visible_history(&self) -> Result<usize, CorpusError> {
        let out = git(&self.root, ["log", "--all", "--format=%H"])?;
        Ok(out.lines().filter(|l| !l.trim().is_empty()).count())
    }
}

/// Build the environment for one task under `dest`.
///
/// `source` is the repository the task was mined from; it is read from and
/// never written to.
pub fn materialize(source: &Path, task: &Task, dest: &Path) -> Result<Sandbox, CorpusError> {
    fs::create_dir_all(dest)?;

    git(dest, ["init", "-q"])?;
    // Identity is required for the commit below and must not depend on
    // whatever the host developer happens to have configured.
    git(dest, ["config", "user.email", "arena@samaritan.invalid"])?;
    git(dest, ["config", "user.name", "Samaritan Arena"])?;

    // Fetch the parent commit and nothing else. `allowAnySHA1InWant` lets us
    // name a commit that is not at the tip of any branch; `protocol.file.allow`
    // is required because newer git refuses file-transport by default.
    git(
        dest,
        [
            "-c",
            "uploadpack.allowAnySHA1InWant=true",
            "-c",
            "protocol.file.allow=always",
            "fetch",
            "-q",
            "--depth",
            "1",
            "--no-tags",
            &source.to_string_lossy(),
            &task.parent,
        ],
    )?;
    git(dest, ["checkout", "-q", "--detach", "FETCH_HEAD"])?;

    // Fetching by URL leaves no remote behind, but a stale config or a future
    // change to git's defaults could. Strip whatever is there.
    for remote in git(dest, ["remote"])?.lines() {
        let r = remote.trim();
        if !r.is_empty() {
            git(dest, ["remote", "remove", r])?;
        }
    }

    // Overlay the commit's tests as plain bytes. This is the step that keeps
    // the solution's objects out: the content crosses as file contents, not as
    // anything git can trace back to a commit.
    for path in &task.test_paths {
        let bytes = git_bytes(source, ["show", &format!("{}:{}", task.commit, path)])?;
        let target = dest.join(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&target, bytes)?;
    }

    // A test file the commit deleted has to go, or the suite fails for a
    // reason that has nothing to do with the task.
    for path in &task.deleted_test_paths {
        let target = dest.join(path);
        if target.exists() {
            fs::remove_file(&target)?;
        }
    }

    // Commit the overlay so the working tree is clean and the agent's own
    // `git diff` shows only its own work.
    git(dest, ["add", "-A"])?;
    git(
        dest,
        [
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "tests for the task",
        ],
    )?;

    let sandbox = Sandbox {
        root: dest.to_path_buf(),
        task_id: task.id.clone(),
        solving_commit: task.commit.clone(),
    };
    sandbox.assert_sealed()?;
    Ok(sandbox)
}
