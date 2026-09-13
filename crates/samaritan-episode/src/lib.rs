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
    /// The full reasoning trace behind a reasoning answer — what verified
    /// self-training (STaR/ReST) learns from when the answer graded correct.
    /// `None` for a coding episode.
    pub trace: Option<String>,
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
        trace: None,
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
    /// The full completion that produced this answer — the reasoning trace and
    /// all. Kept because verified self-training (STaR/ReST) learns from the
    /// *trace* of a correct solve, not just its final answer.
    pub raw: String,
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
You are answering a single hard exam question. Reason carefully but be EFFICIENT: \
do not re-derive or re-verify the same result over and over, and do not pad your \
working. Your output space is limited — if your reasoning is running long, stop \
and commit to your best answer instead of continuing to deliberate, and always \
leave room to finish. End your reply with two lines, exactly:\n\
Answer: <your final answer, as short as the question allows>\n\
Confidence: <a number from 0 to 1>\n\
State a low confidence when unsure. A confident wrong answer is worse than an \
honest low-confidence one — but running out of space with no answer is worst of \
all, so always give the two closing lines.";

/// Instructions for the tool-using solver: the same closing discipline as
/// [`REASONING_SYSTEM`], plus a Python tool. The model reasons, and when it needs
/// to compute — arithmetic, algebra, enumeration, or *checking* a candidate — it
/// emits a fenced ```python block and stops; the harness runs it and feeds the
/// exact output back. This kills the careless-computation error class a fixed base
/// makes most often, and lets it verify an answer before committing.
pub const TOOL_LOOP_SYSTEM: &str = "\
You are answering a single hard exam question, and you have a Python tool.\n\
To run code, output a fenced ```python code block and then STOP writing — you will \
be shown the exact stdout/stderr, then you continue. Use the tool for anything \
mechanical: arithmetic, algebra, enumeration, simulation, and above all to CHECK \
your answer before committing it — do not do heavy computation in your head, that \
is where careless mistakes happen. You may use the tool several times. Reason \
carefully but be EFFICIENT: do not re-derive the same result over and over. When \
you are sure, end your reply with two lines, exactly:\n\
Answer: <your final answer, as short as the question allows>\n\
Confidence: <a number from 0 to 1>\n\
State a low confidence when unsure; a confident wrong answer is worse than an \
honest hedge. Do not write the Answer line in the same turn as a code block — run \
the code, read the output, then answer.";

/// There is no 100 % confidence. 0.95 is the ceiling: it is the highest a
/// calibrated answer may claim, and the most any answer records. A confidence
/// *above* it is not evidence, it is a tell — an overconfident model claims
/// certainty about everything — so it is capped to 0.95 when recorded and
/// triggers a skeptical re-pass. Exactly 0.95 is allowed and taken as-is.
pub const CONFIDENCE_CAP: f64 = 0.95;

/// Whether a stated confidence is high enough to distrust and re-reason.
///
/// Strictly *above* the cap — the operator's rule is "100 %, or higher than
/// 95 %". A calibrated answer at exactly 0.95 (the allowed ceiling, and the value
/// a confident-but-honest model most often gives) is **not** re-thought: doing so
/// spent a whole extra reasoning pass re-deriving a correct answer to land on the
/// same 0.95, which is waste, not diligence.
pub fn should_rethink(confidence: f64) -> bool {
    confidence > CONFIDENCE_CAP
}

/// The recorded confidence, never above the cap.
pub fn cap_confidence(confidence: f64) -> f64 {
    confidence.min(CONFIDENCE_CAP)
}

/// The live solver: the local model, through [`samaritan_agent::Agent`].
///
/// The reasoning counterpart of `impl Decider for Agent`. It runs one
/// completion, lets the model think, and lifts the final answer and confidence
/// out of the reply — tolerant of a `<think>...</think>` block (Qwen3-Thinking
/// and friends emit one) and of the prose a chat model wraps around its answer.
/// It never grades itself; grading is [`Task::grade`]'s, in the episode.
///
/// Confidence discipline: a first pass that comes back at or above
/// [`CONFIDENCE_CAP`] is not trusted — the model is told it was overconfident
/// and made to re-examine its own answer skeptically, and the reconsidered
/// answer is taken. Recorded confidence is always capped, because certainty is
/// not a thing a calibrated solver claims. The cost is a second completion on
/// the over-sure items, which is exactly where the confident-wrong errors — the
/// ones a hard exam punishes most — hide.
impl ReasoningSolver for samaritan_agent::Agent {
    fn solve(&self, question: &str, lessons: &[String]) -> Result<ReasoningAnswer, EpisodeError> {
        let first = answer_once(self, question, lessons, None)?;
        // The winning pass keeps its own trace; a rethink's tokens add to the
        // first pass's so the cost is honest.
        let chosen = if should_rethink(first.confidence) {
            let mut second = answer_once(self, question, lessons, Some(&first.answer))?;
            second.tokens += first.tokens;
            second
        } else {
            first
        };
        Ok(ReasoningAnswer { confidence: cap_confidence(chosen.confidence), ..chosen })
    }
}

/// One reasoning completion through the agent. `reconsider` carries a prior
/// answer to re-examine — the skeptical second pass. A free function rather than
/// a method because `Agent` is defined in another crate.
fn answer_once(
    agent: &samaritan_agent::Agent,
    question: &str,
    lessons: &[String],
    reconsider: Option<&str>,
) -> Result<ReasoningAnswer, EpisodeError> {
    let mut user = question.to_string();
    if !lessons.is_empty() {
        user.push_str(&format!("\n\nKeep in mind:\n- {}", lessons.join("\n- ")));
    }
    if let Some(prior) = reconsider {
        user.push_str(&format!(
            "\n\nYour first answer was: {prior}\nYou were highly confident, but \
             overconfidence is a common failure and certainty is never warranted. \
             Re-examine skeptically — look for an error, a missed case, or a wrong \
             assumption. Then give your final answer and a calibrated confidence no \
             higher than 0.95."
        ));
    }
    let cfg = agent.config();
    let (content, usage) = agent
        .complete(REASONING_SYSTEM, &user, None, None, cfg.temperature, cfg.seed.unwrap_or(0))
        .map_err(|e| EpisodeError::Agent(e.to_string()))?;
    let (answer, confidence) = extract_answer(&content);
    Ok(ReasoningAnswer { answer, confidence, tokens: usage.total(), raw: content })
}

/// A [`ReasoningSolver`] that lets the model call a Python tool mid-reasoning.
///
/// The "reason deeper" path: instead of one shot, the model may run code (bounded
/// by `max_tool_steps`), read the exact output, and continue — decompose, compute,
/// and self-verify before committing. It wraps a live [`Agent`](samaritan_agent::Agent)
/// and drives the same `complete()` transport; because `complete()` is one-shot
/// (no multi-turn), the loop is emulated in a single growing user message that
/// carries the running tool log, which a thinking model handles fine.
///
/// Grading, calibration, and the confidence discipline are unchanged: `solve`
/// applies [`should_rethink`]/[`cap_confidence`] exactly as the single-shot impl
/// does, and returns the same [`ReasoningAnswer`] — with `tokens` **summed across
/// every model call** so the compute budget stays honest, and the tool transcript
/// folded into `raw` so a verified solve is still learnable.
///
/// Confinement is the caller's choice, expressed by which working directory and
/// interpreter are handed in: this shells `python` in a scratch directory via
/// [`samaritan_exec::proc::run`], which on a PathChecked-style host runs the code
/// directly (fast, trusted). Point it only at a model trusted not to emit hostile
/// code, or give it a container-backed working directory.
pub struct ToolLoopSolver<'a> {
    agent: &'a samaritan_agent::Agent,
    /// Hard cap on tool iterations, so a model that never converges still stops.
    max_tool_steps: u32,
    /// The Python interpreter to shell (e.g. `python` / `python3`).
    python_bin: String,
    /// Per-execution wall-clock limit.
    exec_timeout: Duration,
    /// Working directory for the subprocess.
    work_dir: PathBuf,
}

impl<'a> ToolLoopSolver<'a> {
    pub fn new(
        agent: &'a samaritan_agent::Agent,
        max_tool_steps: u32,
        python_bin: impl Into<String>,
        exec_timeout: Duration,
        work_dir: PathBuf,
    ) -> Self {
        Self {
            agent,
            max_tool_steps: max_tool_steps.max(1),
            python_bin: python_bin.into(),
            exec_timeout,
            work_dir,
        }
    }

    /// One tool-using pass. `reconsider` carries a prior answer to re-examine (the
    /// skeptical second pass), mirroring [`answer_once`].
    fn run_loop(
        &self,
        question: &str,
        lessons: &[String],
        reconsider: Option<&str>,
    ) -> Result<ReasoningAnswer, EpisodeError> {
        let mut base = question.to_string();
        if !lessons.is_empty() {
            base.push_str(&format!("\n\nKeep in mind:\n- {}", lessons.join("\n- ")));
        }
        if let Some(prior) = reconsider {
            base.push_str(&format!(
                "\n\nYour first answer was: {prior}\nYou were highly confident, but \
                 overconfidence is a common failure and certainty is never warranted. \
                 Re-examine skeptically — use the Python tool to check it — then give \
                 your final answer and a calibrated confidence no higher than 0.95."
            ));
        }

        let cfg = self.agent.config();
        let mut tool_log = String::new();
        let mut total_tokens = 0u64;
        let mut last = String::new();

        for _ in 0..self.max_tool_steps {
            let user = if tool_log.is_empty() {
                base.clone()
            } else {
                format!("{base}\n\n{tool_log}Give your final Answer and Confidence, or run more Python.")
            };
            let (content, usage) = self
                .agent
                .complete(TOOL_LOOP_SYSTEM, &user, None, None, cfg.temperature, cfg.seed.unwrap_or(0))
                .map_err(|e| EpisodeError::Agent(e.to_string()))?;
            total_tokens += usage.total();
            last = content.clone();

            // Committed a final answer? Take it — don't run more code.
            if has_final_answer(&content) {
                break;
            }
            // Asked to run code? Run it and feed the exact output back.
            match extract_python_block(&content) {
                Some(code) => {
                    let out = proc::run(
                        &self.python_bin,
                        &["-c".to_string(), code.clone()],
                        &self.work_dir,
                        self.exec_timeout,
                        &[],
                    );
                    tool_log.push_str(&format!(
                        "You ran:\n```python\n{}\n```\nOutput:\n{}\n\n",
                        code.trim(),
                        truncate(&format_tool_output(&out), 4000),
                    ));
                }
                // No answer and no code — nothing more to do; grade what's there.
                None => break,
            }
        }

        let (answer, confidence) = extract_answer(&last);
        let raw = if tool_log.is_empty() { last } else { format!("{tool_log}\n{last}") };
        Ok(ReasoningAnswer { answer, confidence, tokens: total_tokens, raw })
    }
}

impl ReasoningSolver for ToolLoopSolver<'_> {
    fn solve(&self, question: &str, lessons: &[String]) -> Result<ReasoningAnswer, EpisodeError> {
        let first = self.run_loop(question, lessons, None)?;
        // Same overconfidence re-pass as the single-shot solver: a claim above the
        // cap is re-examined (with the tool available again), and its tokens add to
        // the first pass so the cost is honest.
        let chosen = if should_rethink(first.confidence) {
            let mut second = self.run_loop(question, lessons, Some(&first.answer))?;
            second.tokens += first.tokens;
            second
        } else {
            first
        };
        Ok(ReasoningAnswer { confidence: cap_confidence(chosen.confidence), ..chosen })
    }
}

/// True once the reply has committed a final answer — a line beginning `answer:`
/// in the post-`</think>` body. The loop stops here rather than running more code.
/// A mid-reasoning mention ("to find the answer:") does not count, because it is
/// not the start of a line.
fn has_final_answer(content: &str) -> bool {
    let body = match content.rfind("</think>") {
        Some(i) => &content[i + "</think>".len()..],
        None => content,
    };
    body.lines().any(|l| {
        let l = l.trim_start_matches(['#', '*', '-', ' ', '\t']).to_lowercase();
        l.starts_with("answer:") || l.starts_with("final answer:")
    })
}

/// Lift the first fenced Python block out of a reply, or `None`. Accepts
/// ```` ```python ````, ```` ```py ````, or a bare ```` ``` ```` fence, and skips
/// fenced blocks in other languages — the tool-call counterpart of the diff
/// lifter the level-3 proposer already uses.
fn extract_python_block(content: &str) -> Option<String> {
    let mut rest = content;
    loop {
        let open = rest.find("```")?;
        let after = &rest[open + 3..];
        let nl = after.find('\n')?;
        let tag = after[..nl].trim().to_lowercase();
        let body = &after[nl + 1..];
        let close = body.find("```")?;
        let code = &body[..close];
        if tag.is_empty() || tag == "python" || tag == "py" {
            return Some(code.to_string());
        }
        rest = &body[close + 3..];
    }
}

/// Render a subprocess result for the model: stdout, any stderr, and the exit
/// status — so it can react to an error as readily as to a value.
fn format_tool_output(out: &proc::Output) -> String {
    let mut s = String::new();
    if !out.stdout.trim().is_empty() {
        s.push_str(out.stdout.trim_end());
        s.push('\n');
    }
    if !out.stderr.trim().is_empty() {
        s.push_str("[stderr] ");
        s.push_str(out.stderr.trim_end());
        s.push('\n');
    }
    s.push_str(&match &out.completion {
        proc::Completion::Exited { code } => format!("[exit {code}]"),
        proc::Completion::TimedOut { after_secs } => format!("[timed out after {after_secs}s]"),
        proc::Completion::Unstartable { detail } => format!("[could not run python: {detail}]"),
    });
    if s.trim().is_empty() { "[no output]".to_string() } else { s }
}

#[cfg(test)]
mod tool_loop_tests {
    use super::{extract_python_block, has_final_answer};

    #[test]
    fn extracts_a_python_fence() {
        let s = "Let me compute.\n```python\nprint(6*7)\n```\nfrom that...";
        assert_eq!(extract_python_block(s).as_deref(), Some("print(6*7)\n"));
    }

    #[test]
    fn accepts_py_and_bare_and_skips_other_languages() {
        assert_eq!(extract_python_block("```py\nx=1\n```").as_deref(), Some("x=1\n"));
        assert_eq!(extract_python_block("```\nz=3\n```").as_deref(), Some("z=3\n"));
        // a json block is skipped in favour of the later python one
        let s = "```json\n{\"a\":1}\n```\nthen\n```python\ny=2\n```";
        assert_eq!(extract_python_block(s).as_deref(), Some("y=2\n"));
    }

    #[test]
    fn no_fence_is_none() {
        assert_eq!(extract_python_block("just prose, no code here"), None);
    }

    #[test]
    fn final_answer_only_on_an_answer_line() {
        assert!(has_final_answer("<think>work</think>\nAnswer: 42\nConfidence: 0.9"));
        // a mid-reasoning mention is not a commitment to stop
        assert!(!has_final_answer("I need to find the answer: let me run code first"));
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
    // think block, the whole reply is. Confidence is scanned over the whole
    // reply, since a model sometimes states it inside the thinking.
    let body = match content.rfind("</think>") {
        Some(i) => &content[i + "</think>".len()..],
        None => content,
    };
    let confidence = parse_confidence(content).unwrap_or(0.5);
    // Prefer the answer from the post-think body; fall back to the whole reply
    // for a model that answered inside its thinking and emitted only a
    // confidence line after.
    let answer = answer_from(body).or_else(|| answer_from(content)).unwrap_or_default();
    (answer, confidence)
}

/// The final answer in a block of text, or `None` if there is nothing usable.
///
/// Handles the shapes a chat model actually produces: `Answer: X` inline,
/// `Answer:` with the value on the next line, and no marker at all (take the
/// last real line). It never returns the `Confidence:` line — the bug the first
/// live run exposed, where a bare `Answer:` sent the fallback onto the trailing
/// confidence line and graded a correct answer as wrong.
fn answer_from(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        let pos = lower.find("answer:").or_else(|| lower.find("final answer:"));
        if let Some(pos) = pos {
            let after = lower[pos..].find(':').map(|c| pos + c + 1).unwrap_or(pos);
            let inline = line[after..].trim();
            if !inline.is_empty() {
                return Some(inline.to_string());
            }
            // `Answer:` with the value on a following line.
            for next in &lines[i + 1..] {
                let t = next.trim();
                if !t.is_empty() && !is_confidence_line(t) {
                    return Some(t.to_string());
                }
            }
        }
    }
    // No marker: the last non-empty line that is not the confidence line.
    lines
        .iter()
        .rev()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !is_confidence_line(l))
        .map(|s| s.to_string())
}

fn is_confidence_line(line: &str) -> bool {
    line.to_lowercase().trim_start_matches(['*', '#', '-', ' ']).starts_with("confidence:")
}

fn parse_confidence(text: &str) -> Option<f64> {
    // The last confidence line wins.
    let mut value = None;
    for line in text.lines() {
        let lower = line.to_lowercase();
        if let Some(pos) = lower.find("confidence:") {
            value = Some(line[pos + "confidence:".len()..].trim().to_string());
        }
    }
    let v = value?;
    // The value may be "0.8", "0.8 (high)", "80%"; take the leading number.
    let token: String = v.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let n: f64 = token.parse().ok()?;
    // A percentage if it came in as 0-100.
    let n = if n > 1.0 { n / 100.0 } else { n };
    Some(n.clamp(0.0, 1.0))
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
    let (ending, correct, confidence, tokens, answer_text, trace) = match answer {
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
            (ending, correct, a.confidence.clamp(0.0, 1.0), a.tokens, Some(a.answer), Some(a.raw))
        }
        Err(e) => (Ending::AgentFailed { detail: e.to_string() }, false, 0.5, 0, None, None),
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
        trace,
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
            Ok(ReasoningAnswer { answer: self.0.into(), confidence: self.1, tokens: 20, raw: self.0.into() })
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
    fn extract_answer_never_returns_the_confidence_line() {
        // The exact shape that broke the first live run: a bare `Answer:` with
        // the value on the next line, then a Confidence line. The old fallback
        // grabbed "Confidence: 1" and graded a correct answer as wrong.
        let (a, c) = extract_answer("<think>...</think>\nAnswer:\nParis\nConfidence: 1");
        assert_eq!(a, "Paris");
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn extract_answer_ignores_a_markdown_confidence_line_in_the_fallback() {
        // No Answer marker, answer in prose, then a decorated confidence line.
        let (a, _) = extract_answer("The capital is Paris.\n**Confidence:** 0.9");
        assert_eq!(a, "The capital is Paris.");
    }

    #[test]
    fn confidence_is_capped_and_high_confidence_triggers_a_rethink() {
        // No 100%: the recorded confidence never exceeds the cap.
        assert_eq!(cap_confidence(1.0), CONFIDENCE_CAP);
        assert_eq!(cap_confidence(0.99), CONFIDENCE_CAP);
        assert!((cap_confidence(0.7) - 0.7).abs() < 1e-9);
        // Strictly above the cap is distrusted and re-reasoned. Exactly 0.95 (the
        // allowed ceiling) is taken as-is — re-thinking it just re-derives a
        // correct answer to the same 0.95. Below is taken.
        assert!(should_rethink(1.0));
        assert!(should_rethink(0.96));
        assert!(!should_rethink(0.95));
        assert!(!should_rethink(0.9));
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
