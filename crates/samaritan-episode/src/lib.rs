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

    #[error("agent: {0}")]
    Agent(String),
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
    /// The answer a reasoning episode produced, for diagnostics and for later
    /// export of reasoning traces. `None` for a coding episode, whose "answer"
    /// is a diff the sandbox already holds.
    pub answer: Option<String>,
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
        answer: None,
    })
}

/// What a reasoning solver returned for one question.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningAnswer {
    /// The answer to grade against the key. The solver's *final* answer, not its
    /// working — the grader matches this string.
    pub answer: String,
    /// Stated confidence in `[0, 1]`, so a reasoning episode feeds calibration
    /// exactly as a coding one does. A solver that will not state a confidence
    /// should return 0.5, not lie.
    pub confidence: f64,
    pub tokens: u64,
}

/// Produces an answer to a reasoning question.
///
/// The reasoning counterpart of [`Decider`]: a seam so the reasoning episode is
/// testable against a scripted solver, and so the W2 tool loop (retrieval, code
/// execution) can be dropped in behind it without changing the episode. The
/// solver never grades itself — grading is the harness's, via [`Task::grade`].
pub trait ReasoningSolver {
    fn solve(&self, question: &str, lessons: &[String]) -> Result<ReasoningAnswer, EpisodeError>;
}

/// Instructions for the live solver.
///
/// It asks for a thinking pass, then a machine-findable final answer and a
/// stated confidence — the confidence is not decoration, it is what the episode
/// records for calibration, and the prompt says plainly that a confident wrong
/// answer is worse than an honest hedge (HLE rewards exactly that).
pub const REASONING_SYSTEM: &str = "\
You are answering a single hard exam question. Reason carefully. Then end your \
reply with two lines, exactly:\n\
Answer: <your final answer, as short as the question allows>\n\
Confidence: <a number from 0 to 1>\n\
State a low confidence when unsure. A confident wrong answer is worse than an \
honest low-confidence one.";

/// The live solver: the local model, through [`samaritan_agent::Agent`].
///
/// The reasoning counterpart of `impl Decider for Agent`. It runs one
/// completion, lets the model think, and lifts the final answer and confidence
/// out of the reply — tolerant of a `<think>...</think>` block (Qwen3-Thinking
/// and friends emit one) and of the prose a chat model wraps around its answer.
/// It never grades itself; grading is [`Task::grade`]'s, in the episode.
impl ReasoningSolver for samaritan_agent::Agent {
    fn solve(&self, question: &str, lessons: &[String]) -> Result<ReasoningAnswer, EpisodeError> {
        let user = if lessons.is_empty() {
            question.to_string()
        } else {
            format!("{}\n\nKeep in mind:\n- {}", question, lessons.join("\n- "))
        };
        let cfg = self.config();
        let (content, usage) = self
            .complete(REASONING_SYSTEM, &user, None, None, cfg.temperature, cfg.seed.unwrap_or(0))
            .map_err(|e| EpisodeError::Agent(e.to_string()))?;
        let (answer, confidence) = extract_answer(&content);
        Ok(ReasoningAnswer { answer, confidence, tokens: usage.total() })
    }
}

/// Lift the final answer and stated confidence out of a reasoning reply.
///
/// Drops a thinking block, prefers an explicit `Answer:` marker, and falls back
/// to the last non-empty line — because a model that ignores the format still
/// usually puts its answer last. Confidence defaults to 0.5 (an honest "unsure")
/// when unstated, never to a flattering high value.
pub fn extract_answer(content: &str) -> (String, f64) {
    // Everything after the last </think> is the answer proper; if there is no
    // think block, the whole reply is.
    let body = match content.rfind("</think>") {
        Some(i) => &content[i + "</think>".len()..],
        None => content,
    };
    let confidence = parse_confidence(body).unwrap_or(0.5);
    let answer = marker_value(body, "answer:")
        .or_else(|| marker_value(body, "final answer:"))
        .unwrap_or_else(|| last_nonempty_line(body))
        .trim()
        .to_string();
    (answer, confidence)
}

/// The text after a case-insensitive `marker` on the last line that carries it.
fn marker_value(text: &str, marker: &str) -> Option<String> {
    let mut found = None;
    for line in text.lines() {
        let lower = line.to_lowercase();
        if let Some(pos) = lower.find(marker) {
            // Take from the original line so the answer's own casing survives.
            found = Some(line[pos + marker.len()..].trim().to_string());
        }
    }
    found.filter(|s| !s.is_empty())
}

fn parse_confidence(text: &str) -> Option<f64> {
    let v = marker_value(text, "confidence:")?;
    // The value may be "0.8", "0.8 (high)", "80%"; take the leading number.
    let token: String = v.trim().chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let n: f64 = token.parse().ok()?;
    // A percentage if it came in as 0-100.
    let n = if n > 1.0 { n / 100.0 } else { n };
    Some(n.clamp(0.0, 1.0))
}

fn last_nonempty_line(text: &str) -> String {
    text.lines()
        .rev()
        .map(|l| l.trim())
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

/// Run one reasoning task to a score.
///
/// The parallel of [`run_episode`] for the reasoning surface: no worktree, no
/// suite, no file edits. The solver answers, the harness grades against the
/// key, and the outcome is the *same* [`EpisodeOutcome`] a coding task produces
/// — same utility, same calibration pair — so every level of the search scores
/// both surfaces through one code path.
pub fn run_reasoning_episode(
    task: &Task,
    solver: &dyn ReasoningSolver,
    ledger: &mut Ledger,
    cfg: &EpisodeConfig,
) -> Result<EpisodeOutcome, EpisodeError> {
    let started = Instant::now();
    if !task.is_reasoning() {
        return Err(EpisodeError::Setup(format!(
            "run_reasoning_episode called on a non-reasoning task {}",
            task.id.0
        )));
    }

    let answer = solver.solve(&task.prompt, &cfg.lessons);
    let (ending, correct, confidence, tokens, answer_text) = match answer {
        Ok(a) => {
            // Grading is the harness's job. `grade` returns Some for a reasoning
            // task; the None arm is unreachable here because of the guard above,
            // but a wrong answer is a clean failure, not an error.
            let correct = task.grade(&a.answer).unwrap_or(false);
            let ending = if correct {
                Ending::Solved { at_step: 1 }
            } else {
                Ending::StepsExhausted
            };
            (ending, correct, a.confidence.clamp(0.0, 1.0), a.tokens, Some(a.answer))
        }
        Err(e) => (Ending::AgentFailed { detail: e.to_string() }, false, 0.5, 0, None),
    };

    // The prediction is the solver's stated confidence against the oracle's
    // verdict — the same calibration signal a coding episode records.
    let predictions = vec![(confidence, correct)];
    ledger.append(
        Actor::System,
        &Event::OutcomeObserved {
            decision: samaritan_dsl::DecisionId::new(),
            resolved: correct,
            oracle: serde_json::json!({ "correct": correct, "domain": task.domain() }),
        },
    )?;

    let components = Components {
        task_success: if correct { 1.0 } else { 0.0 },
        brier: brier(&predictions).unwrap_or(0.0),
        approvals_requested: 0,
        seconds: started.elapsed().as_secs_f64(),
    };
    let utility = EpisodeUtility::clean(&components, &cfg.weights);

    ledger.append(
        Actor::System,
        &Event::EpisodeScored {
            decision: samaritan_dsl::DecisionId::new(),
            utility: utility.clone(),
        },
    )?;

    Ok(EpisodeOutcome {
        task: task.id.clone(),
        ending,
        utility,
        components,
        steps: 1,
        predictions,
        tokens,
        violations: Vec::new(),
        answer: answer_text,
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

#[cfg(test)]
mod reasoning_tests {
    use super::*;
    use samaritan_corpus::{AnswerKind, Split, Task, TaskId};
    use samaritan_ledger::{FixedClock, Ledger};

    /// A solver that returns a fixed answer at a fixed confidence.
    struct Fixed(&'static str, f64);
    impl ReasoningSolver for Fixed {
        fn solve(&self, _q: &str, _l: &[String]) -> Result<ReasoningAnswer, EpisodeError> {
            Ok(ReasoningAnswer { answer: self.0.into(), confidence: self.1, tokens: 20 })
        }
    }

    struct Broken;
    impl ReasoningSolver for Broken {
        fn solve(&self, _q: &str, _l: &[String]) -> Result<ReasoningAnswer, EpisodeError> {
            Err(EpisodeError::Setup("model down".into()))
        }
    }

    fn task() -> Task {
        Task::reasoning(
            TaskId("r1".into()),
            "numina",
            "6 times 7?",
            "42",
            AnswerKind::ExactMatch,
            "math",
            "2026-09-01T00:00:00+00:00",
            Split::Train,
        )
    }

    fn ledger() -> Ledger {
        Ledger::in_memory(Box::new(FixedClock("2026-09-12T00:00:00Z".into()))).unwrap()
    }

    #[test]
    fn a_correct_answer_solves_and_scores_well() {
        let mut l = ledger();
        let out = run_reasoning_episode(&task(), &Fixed("the answer is 42", 0.9), &mut l, &EpisodeConfig::default()).unwrap();
        assert!(out.ending.solved());
        assert_eq!(out.components.task_success, 1.0);
        // Confident and correct → low Brier contribution.
        assert!(out.predictions == vec![(0.9, true)]);
    }

    #[test]
    fn a_wrong_answer_is_a_clean_failure_not_an_error() {
        let mut l = ledger();
        let out = run_reasoning_episode(&task(), &Fixed("43", 0.9), &mut l, &EpisodeConfig::default()).unwrap();
        assert!(!out.ending.solved());
        assert_eq!(out.components.task_success, 0.0);
        assert_eq!(out.predictions, vec![(0.9, false)]);
    }

    #[test]
    fn a_solver_failure_is_recorded_as_agent_failed() {
        let mut l = ledger();
        let out = run_reasoning_episode(&task(), &Broken, &mut l, &EpisodeConfig::default()).unwrap();
        assert!(matches!(out.ending, Ending::AgentFailed { .. }));
        assert_eq!(out.components.task_success, 0.0);
    }

    #[test]
    fn extract_answer_lifts_the_final_answer_past_the_thinking() {
        let reply = "<think>6*7 is 42, let me double check... yes 42.</think>\n\
                     Answer: 42\nConfidence: 0.9";
        let (a, c) = extract_answer(reply);
        assert_eq!(a, "42");
        assert!((c - 0.9).abs() < 1e-9);
    }

    #[test]
    fn extract_answer_defaults_confidence_and_falls_back_to_the_last_line() {
        // No markers at all: take the last non-empty line, hedge the confidence.
        let (a, c) = extract_answer("<think>...</think>\nThe capital is Paris");
        assert_eq!(a, "The capital is Paris");
        assert!((c - 0.5).abs() < 1e-9);
    }

    #[test]
    fn extract_answer_handles_a_percentage_confidence_and_no_think_block() {
        let (a, c) = extract_answer("Answer: C\nConfidence: 80%");
        assert_eq!(a, "C");
        assert!((c - 0.8).abs() < 1e-9);
    }

    #[test]
    fn extract_answer_preserves_answer_casing() {
        let (a, _) = extract_answer("answer: Marie Curie\nconfidence: 0.7");
        assert_eq!(a, "Marie Curie");
    }

    #[test]
    fn a_coding_task_is_refused_by_the_reasoning_path() {
        // Dispatch discipline: the reasoning entry point must not silently score
        // a coding task, whose success is a suite, not a string.
        let mut l = ledger();
        let mut coding = task();
        coding.kind = samaritan_corpus::TaskKind::Coding;
        let r = run_reasoning_episode(&coding, &Fixed("42", 0.9), &mut l, &EpisodeConfig::default());
        assert!(r.is_err());
    }
}
