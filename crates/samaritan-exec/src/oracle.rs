//! The real oracle runner: builds a task state and runs its test suite.
//!
//! Used by corpus screening, which must ask two different questions about the
//! same task — "does the suite pass where the commit fixed it?" and "does it
//! fail before?" — and therefore needs to materialise two *different* states.
//!
//! One of those states necessarily contains the answer. That is fine here and
//! nowhere else: screening happens once, offline, before the corpus is used,
//! and no agent is running. [`SandboxOracle::solution_state`] is named to make
//! that obvious, and an episode must never call it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use samaritan_corpus::git::git;
use samaritan_corpus::screen::{OracleOutcome, OracleRunner};
use samaritan_corpus::{Task, materialize};

use crate::proc::{self, Completion};

/// Runs task oracles in throwaway directories under a scratch root.
pub struct SandboxOracle {
    source: PathBuf,
    scratch: PathBuf,
    counter: u64,
    /// Extra environment for the suite, e.g. `CARGO_TERM_COLOR=never`.
    env: Vec<(String, String)>,
}

impl SandboxOracle {
    pub fn new(source: &Path, scratch: &Path) -> Self {
        Self {
            source: source.to_path_buf(),
            scratch: scratch.to_path_buf(),
            counter: 0,
            env: vec![("CARGO_TERM_COLOR".into(), "never".into())],
        }
    }

    pub fn with_env(mut self, env: Vec<(String, String)>) -> Self {
        self.env = env;
        self
    }

    fn next_dir(&mut self, label: &str) -> PathBuf {
        self.counter += 1;
        self.scratch.join(format!("{label}-{}", self.counter))
    }

    /// Check out the commit that solved the task.
    ///
    /// **Screening only.** This state contains the solution by construction;
    /// handing it to an agent would make the episode meaningless. Episodes use
    /// [`materialize`], which is built so the solution cannot be reached.
    fn solution_state(&mut self, task: &Task) -> Result<PathBuf, String> {
        let dir = self.next_dir("solution");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        git(&dir, ["init", "-q"]).map_err(|e| e.to_string())?;
        git(
            &dir,
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
                &self.source.to_string_lossy(),
                &task.commit,
            ],
        )
        .map_err(|e| e.to_string())?;
        git(&dir, ["checkout", "-q", "--detach", "FETCH_HEAD"]).map_err(|e| e.to_string())?;
        Ok(dir)
    }

    /// The state an agent would actually be given: parent source, new tests.
    fn task_state(&mut self, task: &Task) -> Result<PathBuf, String> {
        let dir = self.next_dir("task");
        let sandbox = materialize(&self.source, task, &dir).map_err(|e| e.to_string())?;
        sandbox.assert_sealed().map_err(|e| e.to_string())?;
        Ok(dir)
    }
}

impl OracleRunner for SandboxOracle {
    fn run(&mut self, task: &Task, at_solution: bool) -> OracleOutcome {
        let dir = if at_solution {
            self.solution_state(task)
        } else {
            self.task_state(task)
        };

        let dir = match dir {
            Ok(d) => d,
            // Could not even build the state. Distinguished from a failing
            // suite: a task whose environment will not assemble is mis-cut,
            // not hard.
            Err(_) => return OracleOutcome::Errored,
        };

        let out = proc::run(
            &task.oracle.program,
            &task.oracle.args,
            &dir,
            Duration::from_secs(task.oracle.timeout_secs),
            &self.env,
        );

        // Best-effort cleanup. On Windows a surviving child can hold a handle
        // and make removal fail, which is why the runner kills process trees
        // rather than single processes; if it still fails, leaving a directory
        // behind is better than failing the screening run.
        let _ = std::fs::remove_dir_all(&dir);

        match out.completion {
            Completion::Exited { code: 0 } => OracleOutcome::Passed,
            Completion::Exited { .. } => OracleOutcome::Failed,
            Completion::TimedOut { .. } => OracleOutcome::TimedOut,
            Completion::Unstartable { .. } => OracleOutcome::Errored,
        }
    }
}
