//! Mining scoreable agent tasks out of real commit history.
//!
//! For each commit that changes both source and tests, there is a task hiding
//! in it: start from the parent, apply only the commit's *test* changes, and
//! the suite now fails exactly where the commit fixed something. The prompt is
//! the commit subject. The oracle is the suite going green. Ground truth is
//! free and nobody had to label anything.
//!
//! Three things make this harder than it sounds, and all three are failures of
//! *measurement* rather than of engineering — they do not crash, they quietly
//! produce numbers that mean nothing:
//!
//! 1. **Leakage.** The answer is sitting in the repository the task was cut
//!    from. See [`sandbox`], which is built so the solution's git objects never
//!    enter the agent's environment at all.
//! 2. **Flakes.** A non-deterministic test contributes noise to every utility
//!    estimate that includes it, and the certificate machinery downstream will
//!    faithfully convert that noise into confident conclusions. See [`screen`].
//! 3. **Contamination of the split.** If the held-out set is chosen at random,
//!    a later commit can teach the agent about an earlier held-out one. The
//!    split is therefore by *time* — see [`Split`].

pub mod git;
pub mod mine;
pub mod sandbox;
pub mod screen;

use serde::{Deserialize, Serialize};

pub use mine::{MineOptions, mine};
pub use sandbox::{Sandbox, materialize};
pub use screen::{OracleOutcome, OracleRunner, ScreenReport, screen_for_flakes};

#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("could not run git: {detail}")]
    GitSpawn { detail: String },

    #[error("git {args} failed ({status}): {stderr}")]
    Git {
        args: String,
        status: i32,
        stderr: String,
    },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("could not encode manifest: {0}")]
    Encode(#[from] serde_json::Error),

    #[error("{0}")]
    Invalid(String),
}

/// Identity of a task: repository name plus the commit that solved it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(pub String);

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// How the oracle is run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleSpec {
    /// Program, e.g. `cargo`.
    pub program: String,
    /// Arguments, e.g. `["test", "--quiet"]`.
    pub args: Vec<String>,
    /// Seconds after which a run counts as failed. A hung suite is a failure,
    /// not a reason to wait forever.
    pub timeout_secs: u64,
}

impl OracleSpec {
    pub fn cargo_test() -> Self {
        Self {
            program: "cargo".into(),
            args: vec!["test".into(), "--quiet".into()],
            timeout_secs: 300,
        }
    }

    pub fn pytest() -> Self {
        Self {
            program: "python".into(),
            args: vec!["-m".into(), "pytest".into(), "-q".into()],
            timeout_secs: 300,
        }
    }
}

/// How a reasoning answer is checked against its key.
///
/// Mirrors the shape the eval grader uses; kept here because the grading rule
/// belongs with the task that defines it, not with the harness that runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnswerKind {
    /// A short free-form answer — a number, a name, a formula.
    ExactMatch,
    /// One option; the key is typically a letter.
    MultipleChoice,
}

/// What kind of task this is, and the data specific to it.
///
/// [`TaskKind::Coding`] is the original: reproduce a commit's test outcome, with
/// the payload in `Task`'s commit/parent/test-path fields and the oracle running
/// a suite. [`TaskKind::Reasoning`] carries its own answer key and is graded by
/// matching the agent's answer — the surface that transfers to HLE. The two are
/// co-equal; a run may train on both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "task_kind", rename_all = "snake_case")]
pub enum TaskKind {
    Coding,
    Reasoning {
        /// The answer key. Never shown to the agent.
        answer: String,
        answer_kind: AnswerKind,
        /// Field, e.g. "math" / "chemistry" / "logic" — for domain routing.
        domain: String,
    },
}

impl Default for TaskKind {
    fn default() -> Self {
        TaskKind::Coding
    }
}

/// Whether `given` answers `expected`, under the answer kind's matching rule.
///
/// A normalised match: lowercase, drop a leading "the answer is" preface, strip
/// punctuation to spaces, collapse whitespace. A single-token key (a number, a
/// choice letter) is accepted anywhere as a standalone token; a multi-word key
/// must appear as a contiguous run. Deliberately conservative — it under-credits
/// an unusual phrasing rather than over-crediting a coincidence. W8 upgrades the
/// reasoning path to an LLM judge; this is the deterministic floor.
pub fn grade_answer(expected: &str, _kind: AnswerKind, given: &str) -> bool {
    let exp = normalize_answer(expected);
    if exp.is_empty() {
        return false;
    }
    let got = normalize_answer(given);
    if got == exp {
        return true;
    }
    let exp_tokens: Vec<&str> = exp.split(' ').filter(|t| !t.is_empty()).collect();
    let got_tokens: Vec<&str> = got.split(' ').filter(|t| !t.is_empty()).collect();
    if exp_tokens.len() == 1 {
        got_tokens.contains(&exp_tokens[0])
    } else {
        got_tokens.windows(exp_tokens.len()).any(|w| w == exp_tokens.as_slice())
    }
}

fn normalize_answer(s: &str) -> String {
    let lower = s.trim().to_lowercase();
    let lower = lower
        .strip_prefix("the answer is")
        .or_else(|| lower.strip_prefix("answer:"))
        .or_else(|| lower.strip_prefix("answer is"))
        .unwrap_or(&lower);
    let mut out = String::with_capacity(lower.len());
    let mut last_space = false;
    for c in lower.chars() {
        if c.is_alphanumeric() {
            out.push(c);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
}

/// Rough size of the change the agent has to reproduce.
///
/// Used to stratify sampling. Without it a level-0 batch drawn at random is
/// mostly one-line fixes, and a policy that improves only on those will look
/// like a policy that improved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Difficulty {
    pub source_files_changed: u32,
    pub lines_added: u32,
    pub lines_removed: u32,
    pub test_files_changed: u32,
}

impl Difficulty {
    /// A single ordinal for stratification. Deliberately crude: it only has to
    /// separate trivial from substantial, not rank two similar commits.
    pub fn magnitude(&self) -> u32 {
        self.lines_added + self.lines_removed + 10 * self.source_files_changed
    }

    /// Coarse bucket used when drawing balanced batches.
    pub fn bucket(&self) -> Bucket {
        match self.magnitude() {
            0..=20 => Bucket::Trivial,
            21..=100 => Bucket::Small,
            101..=400 => Bucket::Medium,
            _ => Bucket::Large,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bucket {
    Trivial,
    Small,
    Medium,
    Large,
}

impl Bucket {
    pub const ALL: [Bucket; 4] = [
        Bucket::Trivial,
        Bucket::Small,
        Bucket::Medium,
        Bucket::Large,
    ];
}

/// Which part of the experiment a task belongs to.
///
/// The train/held-out split is by commit time, never at random. A randomly
/// held-out task can be taught by a *later* training commit that touches the
/// same code, which contaminates the held-out set in a way that is invisible
/// afterwards and inflates every result computed from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    /// Everything strictly before the cutoff. What the search may see.
    Train,
    /// Everything at or after it. The absolute yardstick.
    HeldOut,
    /// The frontier set: tasks believed unreachable by the base scaffold.
    ///
    /// Never trained on and never part of any utility the search optimises.
    /// The moment a frontier task enters the objective it stops being a
    /// frontier and becomes an expensive training task.
    ///
    /// Frontier tasks are scored differently from everything else. Most of
    /// them return zero for a long time and then one flips, so the meaningful
    /// measurement is *time to first solve* rather than mean utility, and
    /// tasks never solved are right-censored observations rather than zeros.
    Frontier,
}

/// Evidence that a frontier task is actually reachable.
///
/// Without one, a frontier task is indistinguishable from an impossible task,
/// and the two produce identical data: no solve, ever. That makes a null
/// result uninterpretable — you cannot tell "self-improvement did not happen"
/// from "there was nothing here to find" — which is the single most likely way
/// for a frontier experiment to waste a month.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolvabilityWitness {
    pub kind: WitnessKind,
    /// What was actually observed, in enough detail to re-check.
    pub evidence: String,
    pub recorded_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessKind {
    /// A stronger model solved it under the same scaffold.
    StrongerModel,
    /// The run's own model solved it once given a hint it will not receive.
    SolvedWithHint,
    /// A person solved it under the same scaffold and constraints.
    Human,
    /// Solved by the same model with a scaffold change we know is reachable
    /// through the mutation grammar. The strongest witness available, because
    /// it names the specific self-modification that would unlock it.
    ReachableScaffold { mutation_sketch: String },
}

/// One task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    /// Logical corpus name, e.g. `ripgrep` or `rust-unseen`.
    pub corpus: String,
    /// The commit that solved it.
    pub commit: String,
    /// The state the agent starts from.
    pub parent: String,
    /// The commit *subject* only.
    ///
    /// The body is withheld deliberately: commit bodies routinely describe the
    /// fix, sometimes in enough detail to reconstruct it, which would make the
    /// task a reading-comprehension exercise instead of a coding one.
    pub prompt: String,
    /// Test files the commit changed. These are applied to the parent.
    pub test_paths: Vec<String>,
    /// Test files the commit deleted, removed from the parent.
    pub deleted_test_paths: Vec<String>,
    /// Source files the commit changed. Recorded for difficulty and analysis;
    /// never shown to the agent.
    pub source_paths: Vec<String>,
    pub oracle: OracleSpec,
    pub difficulty: Difficulty,
    /// Committer date, RFC 3339. The axis the split is taken on.
    pub committed_at: String,
    pub split: Split,
    /// Required for [`Split::Frontier`], meaningless otherwise.
    #[serde(default)]
    pub witness: Option<SolvabilityWitness>,
    /// Coding (the commit/test payload above) or reasoning (an answer key).
    /// Defaults to `Coding` so every task mined or serialized before the
    /// reasoning surface existed still reads correctly.
    #[serde(default)]
    pub kind: TaskKind,
}

impl Task {
    /// A reasoning task: a question graded against an answer key, with no
    /// commit, no repo, and no suite. The coding-only fields are left empty and
    /// are never read for this kind.
    #[allow(clippy::too_many_arguments)]
    pub fn reasoning(
        id: TaskId,
        corpus: impl Into<String>,
        prompt: impl Into<String>,
        answer: impl Into<String>,
        answer_kind: AnswerKind,
        domain: impl Into<String>,
        committed_at: impl Into<String>,
        split: Split,
    ) -> Self {
        Task {
            id,
            corpus: corpus.into(),
            commit: String::new(),
            parent: String::new(),
            prompt: prompt.into(),
            test_paths: Vec::new(),
            deleted_test_paths: Vec::new(),
            source_paths: Vec::new(),
            // A reasoning task's oracle is answer-grading, not a command; this
            // spec is inert for it and the episode dispatches on `kind`.
            oracle: OracleSpec { program: String::new(), args: Vec::new(), timeout_secs: 0 },
            difficulty: Difficulty {
                source_files_changed: 0,
                lines_added: 0,
                lines_removed: 0,
                test_files_changed: 0,
            },
            committed_at: committed_at.into(),
            split,
            witness: None,
            kind: TaskKind::Reasoning {
                answer: answer.into(),
                answer_kind,
                domain: domain.into(),
            },
        }
    }

    /// Whether this is a reasoning task.
    pub fn is_reasoning(&self) -> bool {
        matches!(self.kind, TaskKind::Reasoning { .. })
    }

    /// Grade a candidate answer against a reasoning task's key.
    ///
    /// `None` for a coding task — it has no answer key, and a caller that grades
    /// one has dispatched on the wrong thing. `Some(correct)` for a reasoning
    /// task.
    pub fn grade(&self, given: &str) -> Option<bool> {
        match &self.kind {
            TaskKind::Coding => None,
            TaskKind::Reasoning { answer, answer_kind, .. } => {
                Some(grade_answer(answer, *answer_kind, given))
            }
        }
    }

    /// The task's domain, for routing. Empty for coding tasks.
    pub fn domain(&self) -> &str {
        match &self.kind {
            TaskKind::Coding => "",
            TaskKind::Reasoning { domain, .. } => domain,
        }
    }
}

/// A named set of tasks from one repository.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Corpus {
    pub name: String,
    /// Absolute path to the source repository the tasks were mined from.
    pub repo_path: String,
    /// RFC 3339 cutoff separating [`Split::Train`] from [`Split::HeldOut`].
    pub cutoff: String,
    /// Whether any arm is permitted to train on this corpus.
    ///
    /// `false` marks a **transfer** corpus: mined from repositories no run
    /// touches, used only to ask whether capability generalises. Marking it in
    /// the data rather than in a runbook is the point — the training loop can
    /// assert on it.
    pub trainable: bool,
    pub tasks: Vec<Task>,
}

impl Corpus {
    pub fn tasks_in(&self, split: Split) -> impl Iterator<Item = &Task> {
        self.tasks.iter().filter(move |t| t.split == split)
    }

    /// Tasks of one split grouped into difficulty buckets, for stratified
    /// sampling.
    pub fn stratified(&self, split: Split) -> std::collections::BTreeMap<Bucket, Vec<&Task>> {
        let mut m: std::collections::BTreeMap<Bucket, Vec<&Task>> = Default::default();
        for t in self.tasks_in(split) {
            m.entry(t.difficulty.bucket()).or_default().push(t);
        }
        m
    }

    /// Every task the training loop is allowed to look at.
    ///
    /// Returns empty for a transfer corpus regardless of the split, so a
    /// caller that forgets to check `trainable` gets nothing rather than
    /// silently contaminating the experiment. Frontier tasks are excluded
    /// structurally — [`Split::Train`] is the only split this can ever
    /// return, so there is no argument by which a frontier task reaches the
    /// objective.
    pub fn trainable_tasks(&self) -> Vec<&Task> {
        if !self.trainable {
            return Vec::new();
        }
        self.tasks_in(Split::Train).collect()
    }

    /// Every frontier task must carry a solvability witness.
    ///
    /// Enforced rather than documented, because the cost of getting it wrong
    /// is only discovered at the end of a run: an unsolved unwitnessed task
    /// produces exactly the same data as an unsolved witnessed one, and only
    /// the second is evidence of anything.
    pub fn check_frontier_witnessed(&self) -> Result<(), CorpusError> {
        for t in self.tasks_in(Split::Frontier) {
            if t.witness.is_none() {
                return Err(CorpusError::Invalid(format!(
                    "frontier task {} has no solvability witness; a failure to solve it                      would be uninterpretable",
                    t.id
                )));
            }
        }
        Ok(())
    }
}

/// The full set of corpora a run was configured with.
///
/// Hashed into [`Event::RunConfigured`]'s `corpus_manifest` so that a later
/// claim about transfer can be checked against what the run actually saw.
///
/// [`Event::RunConfigured`]: https://docs.rs/samaritan-ledger
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub corpora: Vec<Corpus>,
}

impl Manifest {
    pub fn digest(&self) -> samaritan_dsl::Digest {
        samaritan_dsl::hash_json(self)
    }

    /// Corpora no arm may train on.
    pub fn transfer_corpora(&self) -> impl Iterator<Item = &Corpus> {
        self.corpora.iter().filter(|c| !c.trainable)
    }

    /// Fail loudly if any repository appears as both trainable and transfer.
    ///
    /// The single most damaging way to get this wrong, and the easiest to do
    /// by accident when adding a corpus: transfer numbers from a repository
    /// the agent trained on are not transfer numbers, and nothing downstream
    /// can detect it.
    pub fn check_disjoint(&self) -> Result<(), CorpusError> {
        use std::collections::HashSet;
        let trained: HashSet<&str> = self
            .corpora
            .iter()
            .filter(|c| c.trainable)
            .map(|c| c.repo_path.as_str())
            .collect();
        for c in self.corpora.iter().filter(|c| !c.trainable) {
            if trained.contains(c.repo_path.as_str()) {
                return Err(CorpusError::Invalid(format!(
                    "repository {} is used for both training and transfer; \
                     transfer results from it would be meaningless",
                    c.repo_path
                )));
            }
        }
        Ok(())
    }
}

/// Rules for deciding whether a path is a test.
///
/// Heuristic, and it will misclassify something eventually. It errs toward
/// calling a file a test: a source file wrongly treated as a test gets handed
/// to the agent as part of the specification, which makes one task too easy.
/// A test wrongly treated as source is never applied, so the suite does not
/// fail, and the task is silently dropped for having no failing oracle —
/// wasteful but not corrupting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestPatterns {
    /// Path segments that mark a directory of tests.
    pub dir_segments: Vec<String>,
    /// Filename prefixes, e.g. `test_`.
    pub file_prefixes: Vec<String>,
    /// Filename infixes before the extension, e.g. `_test`, `.test`, `.spec`.
    pub file_stems: Vec<String>,
}

impl Default for TestPatterns {
    fn default() -> Self {
        Self {
            dir_segments: ["tests", "test", "__tests__", "spec", "testdata"]
                .map(String::from)
                .to_vec(),
            file_prefixes: ["test_"].map(String::from).to_vec(),
            file_stems: ["_test", ".test", ".spec", "_spec"]
                .map(String::from)
                .to_vec(),
        }
    }
}

impl TestPatterns {
    pub fn is_test(&self, path: &str) -> bool {
        let norm = path.replace('\\', "/");
        let segments: Vec<&str> = norm.split('/').collect();

        if segments
            .iter()
            .take(segments.len().saturating_sub(1))
            .any(|s| self.dir_segments.iter().any(|d| d == s))
        {
            return true;
        }

        let Some(file) = segments.last() else {
            return false;
        };
        if self.file_prefixes.iter().any(|p| file.starts_with(p)) {
            return true;
        }

        let stem = file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file);
        self.file_stems.iter().any(|s| stem.ends_with(s))
    }
}
