//! Screening flaky and mis-cut tasks out of a corpus before it is used.
//!
//! A non-deterministic test does not merely add noise. Downstream, the
//! certificate machinery converts observed differences into confident
//! statistical claims, and it cannot distinguish a difference caused by a
//! better policy from one caused by a test that fails a third of the time. The
//! anytime-valid guarantee is about sampling error under a fixed distribution;
//! a flaky oracle violates that assumption rather than being absorbed by it.
//!
//! So every candidate task is checked twice before admission, against two
//! separate requirements:
//!
//! - At the **solving commit** the suite must pass, twice, identically.
//!   Otherwise the task has no reliable success condition.
//! - At the **parent with new tests** the suite must fail. Otherwise there is
//!   nothing for the agent to do and the task is free marks.
//!
//! The second check catches more than it sounds like. It rejects commits whose
//! tests never actually exercised the fix, commits where our test/source
//! classification guessed wrong, and tasks whose environment cannot build at
//! all.

use serde::{Deserialize, Serialize};

use crate::Task;

/// What happened when the oracle ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OracleOutcome {
    Passed,
    Failed,
    /// Did not finish inside the task's timeout. Treated as a failure for
    /// scoring, but distinguished here because a task that times out during
    /// screening is usually mis-cut rather than genuinely hard.
    TimedOut,
    /// Could not run at all: missing toolchain, unbuildable tree.
    Errored,
}

impl OracleOutcome {
    pub fn passed(self) -> bool {
        self == OracleOutcome::Passed
    }
}

/// Runs a task's oracle somewhere.
///
/// A trait so screening can be tested against scripted outcomes. Process
/// execution policy belongs to the executor, not to the corpus miner, and
/// keeping it behind this boundary means the screening *logic* — which is
/// where the subtle mistakes live — is testable without running anything.
pub trait OracleRunner {
    /// Run the oracle for `task` at the given commit state.
    ///
    /// `at_solution` selects which state to build: the solving commit, or the
    /// parent with the commit's tests applied.
    fn run(&mut self, task: &Task, at_solution: bool) -> OracleOutcome;
}

/// Why a task was rejected, or that it was admitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Screening {
    Admitted,
    /// The suite did not pass at the commit that supposedly fixed it.
    SolutionDoesNotPass { outcome: OracleOutcome },
    /// Two runs at the solving commit disagreed.
    Flaky {
        first: OracleOutcome,
        second: OracleOutcome,
    },
    /// The suite already passed before the fix, so there is no task here.
    NoFailingOracle,
    /// The environment could not run the oracle at all.
    Unrunnable { outcome: OracleOutcome },
}

impl Screening {
    pub fn admitted(&self) -> bool {
        matches!(self, Screening::Admitted)
    }
}

/// The outcome of screening a whole corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenReport {
    pub admitted: Vec<crate::TaskId>,
    pub rejected: Vec<(crate::TaskId, Screening)>,
}

impl ScreenReport {
    pub fn admission_rate(&self) -> f64 {
        let total = self.admitted.len() + self.rejected.len();
        if total == 0 {
            return 0.0;
        }
        self.admitted.len() as f64 / total as f64
    }

    /// How many tasks were rejected for each reason, for reporting.
    ///
    /// Worth watching: a sharp change in the mix usually means the
    /// test-classification heuristic has met a repository it does not
    /// understand, not that the repository is unusual.
    pub fn rejection_breakdown(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut m = std::collections::BTreeMap::new();
        for (_, s) in &self.rejected {
            let key = match s {
                Screening::Admitted => "admitted",
                Screening::SolutionDoesNotPass { .. } => "solution_does_not_pass",
                Screening::Flaky { .. } => "flaky",
                Screening::NoFailingOracle => "no_failing_oracle",
                Screening::Unrunnable { .. } => "unrunnable",
            };
            *m.entry(key).or_insert(0) += 1;
        }
        m
    }
}

/// Screen one task.
pub fn screen_one(runner: &mut dyn OracleRunner, task: &Task) -> Screening {
    // Does the suite pass where it is supposed to?
    let first = runner.run(task, true);
    if matches!(first, OracleOutcome::Errored) {
        return Screening::Unrunnable { outcome: first };
    }
    if !first.passed() {
        return Screening::SolutionDoesNotPass { outcome: first };
    }

    // Same state, again. Disagreement here is non-determinism, and the task is
    // unusable however hard it would otherwise have been.
    let second = runner.run(task, true);
    if second != first {
        return Screening::Flaky { first, second };
    }

    // And it must genuinely fail before the fix, or the agent gets it free.
    let before = runner.run(task, false);
    if matches!(before, OracleOutcome::Errored) {
        return Screening::Unrunnable { outcome: before };
    }
    if before.passed() {
        return Screening::NoFailingOracle;
    }

    Screening::Admitted
}

/// Screen a corpus in place, dropping every task that does not qualify.
pub fn screen_for_flakes(
    runner: &mut dyn OracleRunner,
    corpus: &mut crate::Corpus,
) -> ScreenReport {
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();
    let mut keep = Vec::new();

    for task in std::mem::take(&mut corpus.tasks) {
        match screen_one(runner, &task) {
            Screening::Admitted => {
                admitted.push(task.id.clone());
                keep.push(task);
            }
            other => rejected.push((task.id.clone(), other)),
        }
    }

    corpus.tasks = keep;
    ScreenReport { admitted, rejected }
}
