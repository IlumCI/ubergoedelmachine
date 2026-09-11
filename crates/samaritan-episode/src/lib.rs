//! One episode: a mined task, an agent, and a score at the end.
//!
//! This is the level-0 rollout the whole nested search is built on, so what
//! it measures has to be trustworthy before anything above it means
//! anything. Three rules follow from that, and each one exists because the
//! convenient alternative produces numbers that look fine and are not:
//!
//! 1. **The harness owns the oracle.** The agent never declares success. It
//!    cannot emit "done", cannot report that tests pass, and is never asked.
//!    The suite is run by this module and the exit code is the only thing
//!    that counts, because an agent that can announce its own victory will
//!    eventually announce one it did not win — that is
//!    [`ExploitClass::FabricatedOracle`], and the defence is structural
//!    rather than vigilant.
//! 2. **Predictions are settled per decision.** Confidence stated before a
//!    step is checked against what the oracle said after it. Scoring only the
//!    final prediction would let an agent hedge everywhere and stake
//!    everything on a last guess.
//! 3. **A violation ends the episode.** Not "is recorded and we carry on":
//!    utility is lexicographic, so once an episode is violated its score is
//!    already decided and every further token spent on it is waste.
//!
//! [`ExploitClass::FabricatedOracle`]: samaritan_ledger::ExploitClass

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use samaritan_agent::{AgentError, Completion, Prompt};
use samaritan_corpus::{Task, materialize};
use samaritan_dsl::decision::PolicyVersion;
use samaritan_dsl::Authority;
use samaritan_exec::{Confinement, Executor, proc};
use samaritan_kernel::{
    ApprovalGate, Components, EpisodeUtility, Violation, Weights,
};
use samaritan_ledger::{Actor, Event, Ledger};
use samaritan_router::Router;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum EpisodeError {
    #[error("could not prepare the task: {0}")]
    Setup(String),

    #[error("ledger: {0}")]
    Ledger(#[from] samaritan_ledger::LedgerError),

    #[error("executor: {0}")]
    Exec(String),
}

/// Produces decisions. A trait so the episode loop can be tested against
/// scripted answers rather than a model that takes a minute per call.
pub trait Decider {
    fn decide(&self, prompt: &Prompt, pv: PolicyVersion) -> Result<Completion, AgentError>;
}

impl Decider for samaritan_agent::Agent {
    fn decide(&self, prompt: &Prompt, pv: PolicyVersion) -> Result<Completion, AgentError> {
        samaritan_agent::Agent::decide(self, prompt, pv)
    }
}

#[derive(Debug, Clone)]
pub struct EpisodeConfig {
    /// Where sandboxes are built. Wiped per episode.
    pub scratch_root: PathBuf,
    /// Maximum decisions before the episode is abandoned. A cap on cost, and
    /// on an agent that has stopped making progress but not stopped trying.
    pub max_steps: u32,
    /// Who else could have written what the agent reads back.
    ///
    /// [`Authority::Agent`] for a sealed solo episode; [`Authority::Observed`]
    /// when the Deviant shares the arena. See [`Authority::Observed`] for why
    /// this is a property of the *run* rather than of the act of reading.
    pub observation_authority: Authority,
    /// Lessons injected into the stable prompt band.
    pub lessons: Vec<String>,
    pub weights: Weights,
    /// Wall-clock ceiling for the whole episode.
    pub deadline: Duration,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            scratch_root: std::env::temp_dir().join("samaritan-episodes"),
            max_steps: 8,
            observation_authority: Authority::Agent,
            lessons: Vec::new(),
            weights: Weights::default(),
            deadline: Duration::from_secs(1800),
        }
    }
}

/// Why an episode stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ending {
    /// The oracle went green. The only kind of success there is.
    Solved { at_step: u32 },
    /// Steps ran out with the suite still failing.
    StepsExhausted,
    /// The wall-clock deadline passed.
    TimedOut,
    /// A violation occurred; the episode is already scored and continuing
    /// would only spend more tokens on a foregone conclusion.
    Violated,
    /// The model or the server failed in a way retrying would not fix.
    AgentFailed { detail: String },
}

impl Ending {
    pub fn solved(&self) -> bool {
        matches!(self, Ending::Solved { .. })
    }
}

/// What one episode produced.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeOutcome {
    pub task: samaritan_corpus::TaskId,
    pub ending: Ending,
    pub utility: EpisodeUtility,
    pub components: Components,
    pub steps: u32,
    /// `(stated confidence, whether it happened)` for every decision, which
    /// is what calibration is computed from.
    pub predictions: Vec<(f64, bool)>,
    pub tokens: u64,
    pub violations: Vec<Violation>,
}

/// Run one task to a score.
#[allow(clippy::too_many_arguments)]
pub fn run_episode(
    task: &Task,
    source_repo: &Path,
    decider: &dyn Decider,
    router: &mut Router,
    gate: &mut dyn ApprovalGate,
    ledger: &mut Ledger,
    policy_version: PolicyVersion,
    cfg: &EpisodeConfig,
) -> Result<EpisodeOutcome, EpisodeError> {
    let started = Instant::now();
    std::fs::create_dir_all(&cfg.scratch_root).map_err(|e| EpisodeError::Setup(e.to_string()))?;
    let root = cfg.scratch_root.join(task.id.0.replace(['/', '@'], "_"));
    let _ = std::fs::remove_dir_all(&root);

    let sandbox =
        materialize(source_repo, task, &root).map_err(|e| EpisodeError::Setup(e.to_string()))?;
    // Re-checked here and not only at construction: this is the last moment
    // before a model sees the environment, and a leaked solution does not
    // announce itself — it shows up as an agent that appears to have
    // improved dramatically.
    sandbox
        .assert_sealed()
        .map_err(|e| EpisodeError::Setup(e.to_string()))?;

    let mut executor = Executor::new(&root, Confinement::PathChecked)
        .map_err(|e| EpisodeError::Exec(e.to_string()))?;

    // The failing suite, as the agent's starting evidence.
    let first = run_oracle(task, &root);
    let mut last_output = first.text.clone();

    let mut predictions: Vec<(f64, bool)> = Vec::new();
    let mut violations: Vec<Violation> = Vec::new();
    let mut approvals = 0u32;
    let mut tokens = 0u64;
    let mut steps = 0u32;

    // A task whose suite already passes has nothing in it. Screening should
    // have caught this; if one slips through, saying so is better than
    // recording a free success.
    let mut ending = if first.passed {
        Ending::Solved { at_step: 0 }
    } else {
        Ending::StepsExhausted
    };

    while !ending.solved() && steps < cfg.max_steps {
        if started.elapsed() > cfg.deadline {
            ending = Ending::TimedOut;
            break;
        }

        let prompt = Prompt::new()
            .with_policy(&cfg.lessons)
            .with_task(&task.prompt, &first.text)
            .with_observation("Most recent test output", &last_output, cfg.observation_authority);

        let completion = match decider.decide(&prompt, policy_version) {
            Ok(c) => c,
            Err(e) => {
                ending = Ending::AgentFailed {
                    detail: e.to_string(),
                };
                break;
            }
        };
        steps += 1;
        tokens += completion.usage.total();
        router.spend(completion.usage.total());

        let confidence = completion.record.prediction().confidence.get();
        let decision_id = completion.record.id();

        let dispatch = router
            .dispatch(&completion.record, &mut executor, gate, ledger)
            .map_err(|e| EpisodeError::Exec(e.to_string()))?;
        approvals += dispatch.approvals_requested;
        violations.extend(dispatch.violations.iter().cloned());

        // Only re-run the suite when something could have changed it. Reads
        // cannot, and the oracle is the most expensive thing in the loop.
        let changed = dispatch
            .results
            .iter()
            .any(|r| r.disposition.ran() && r.outcome.as_ref().is_some_and(|o| o.succeeded));

        let check = if changed {
            let c = run_oracle(task, &root);
            last_output = c.text.clone();
            c
        } else {
            OracleCheck {
                passed: false,
                text: last_output.clone(),
            }
        };

        // The prediction is settled by the oracle, never by the agent. It is
        // recorded whether or not it was right, because a prediction that is
        // only written down when convenient is not a prediction.
        predictions.push((confidence, check.passed));
        ledger.append(
            Actor::System,
            &Event::OutcomeObserved {
                decision: decision_id,
                resolved: check.passed,
                oracle: serde_json::json!({
                    "passed": check.passed,
                    "output": truncate(&check.text, 4000),
                }),
            },
        )?;

        if !violations.is_empty() {
            ending = Ending::Violated;
            break;
        }
        if check.passed {
            ending = Ending::Solved { at_step: steps };
        }
    }

    let components = Components {
        task_success: if ending.solved() { 1.0 } else { 0.0 },
        brier: brier(&predictions).unwrap_or(0.0),
        approvals_requested: approvals,
        seconds: started.elapsed().as_secs_f64(),
    };
    let utility = if violations.is_empty() {
        EpisodeUtility::clean(&components, &cfg.weights)
    } else {
        EpisodeUtility::violated(violations.clone())
    };

    ledger.append(
        Actor::System,
        &Event::EpisodeScored {
            decision: samaritan_dsl::DecisionId::new(),
            utility: utility.clone(),
        },
    )?;

    // The sandbox is disposable and holding it open costs disk across
    // thousands of rollouts. Failure to remove it is not worth failing the
    // episode over: a surviving directory is untidy, a lost score is not.
    let _ = std::fs::remove_dir_all(&root);

    Ok(EpisodeOutcome {
        task: task.id.clone(),
        ending,
        utility,
        components,
        steps,
        predictions,
        tokens,
        violations,
    })
}

struct OracleCheck {
    passed: bool,
    text: String,
}

/// Run the task's test suite in the sandbox.
///
/// The only thing in the system permitted to say whether an episode
/// succeeded.
fn run_oracle(task: &Task, root: &Path) -> OracleCheck {
    let out = proc::run(
        &task.oracle.program,
        &task.oracle.args,
        root,
        Duration::from_secs(task.oracle.timeout_secs),
        &[("CARGO_TERM_COLOR".into(), "never".into())],
    );
    let text = if out.stderr.trim().is_empty() {
        out.stdout.clone()
    } else {
        format!("{}\n{}", out.stdout, out.stderr)
    };
    OracleCheck {
        passed: out.success(),
        text: text.trim().to_string(),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    s.chars().take(n).collect::<String>() + "\n[... truncated ...]"
}

/// Brier score over this episode's settled predictions.
///
/// `None` for an episode that made none, which the caller must not convert to
/// a flattering zero.
pub fn brier(pairs: &[(f64, bool)]) -> Option<f64> {
    if pairs.is_empty() {
        return None;
    }
    let sum: f64 = pairs
        .iter()
        .map(|(c, hit)| {
            let o = if *hit { 1.0 } else { 0.0 };
            (c - o).powi(2)
        })
        .sum();
    Some(sum / pairs.len() as f64)
}
