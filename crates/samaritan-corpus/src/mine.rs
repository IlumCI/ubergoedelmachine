//! Walking a repository's history and cutting tasks out of it.

use std::path::Path;

use crate::git::git;
use crate::{
    Corpus, CorpusError, Difficulty, OracleSpec, Split, Task, TaskId, TaskKind, TestPatterns,
};

/// Field separator for `git log --format`. A control character, because commit
/// subjects contain every printable delimiter anyone has ever chosen.
const SEP: &str = "\u{1f}";

#[derive(Debug, Clone)]
pub struct MineOptions {
    /// Logical name for the corpus.
    pub corpus_name: String,
    /// RFC 3339 cutoff: commits at or after this are [`Split::HeldOut`].
    pub cutoff: String,
    /// Whether the training loop may use this corpus at all.
    pub trainable: bool,
    pub oracle: OracleSpec,
    pub patterns: TestPatterns,
    /// Stop after this many tasks. `None` walks the whole history.
    pub limit: Option<usize>,
    /// Git revision range to walk, e.g. `HEAD` or `v1.0..HEAD`.
    pub rev_range: String,
}

impl MineOptions {
    pub fn new(corpus_name: impl Into<String>, cutoff: impl Into<String>) -> Self {
        Self {
            corpus_name: corpus_name.into(),
            cutoff: cutoff.into(),
            trainable: true,
            oracle: OracleSpec::cargo_test(),
            patterns: TestPatterns::default(),
            limit: None,
            rev_range: "HEAD".into(),
        }
    }

    /// Mark this corpus as transfer-only: mined for measurement, never trained
    /// on.
    pub fn transfer_only(mut self) -> Self {
        self.trainable = false;
        self
    }
}

/// One commit's worth of raw facts, before we decide whether it is a task.
struct Commit {
    sha: String,
    parents: Vec<String>,
    subject: String,
    committed_at: String,
}

/// Walk `repo`'s history and return every commit that yields a task.
///
/// A commit qualifies when it changes at least one test file and at least one
/// source file. Commits that touch only tests have no bug to fix; commits that
/// touch only source have no oracle to settle them.
pub fn mine(repo: &Path, opts: &MineOptions) -> Result<Corpus, CorpusError> {
    let log = git(
        repo,
        [
            "log",
            "--reverse",
            "--no-merges",
            &format!("--format=%H{SEP}%P{SEP}%cI{SEP}%s"),
            &opts.rev_range,
        ],
    )?;

    let mut tasks = Vec::new();

    for line in log.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(4, SEP).collect();
        if parts.len() < 4 {
            continue;
        }
        let c = Commit {
            sha: parts[0].to_string(),
            parents: parts[1].split_whitespace().map(String::from).collect(),
            committed_at: parts[2].to_string(),
            subject: parts[3].trim().to_string(),
        };

        // A root commit has no "before" state, so there is no task in it.
        // Merges are excluded at the log level: their diff against a single
        // parent is not the change anyone actually made.
        if c.parents.len() != 1 {
            continue;
        }

        if let Some(task) = task_from_commit(repo, &c, opts)? {
            tasks.push(task);
            if let Some(limit) = opts.limit {
                if tasks.len() >= limit {
                    break;
                }
            }
        }
    }

    Ok(Corpus {
        name: opts.corpus_name.clone(),
        repo_path: repo.to_string_lossy().into_owned(),
        cutoff: opts.cutoff.clone(),
        trainable: opts.trainable,
        tasks,
    })
}

fn task_from_commit(
    repo: &Path,
    c: &Commit,
    opts: &MineOptions,
) -> Result<Option<Task>, CorpusError> {
    let parent = &c.parents[0];

    // name-status gives us adds and deletes, which matter: a test file the
    // commit *deleted* has to be deleted in the sandbox too, or the suite
    // fails for a reason unrelated to the task.
    let status = git(
        repo,
        [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-M",
            &c.sha,
        ],
    )?;

    let mut test_paths = Vec::new();
    let mut deleted_test_paths = Vec::new();
    let mut source_paths = Vec::new();

    for line in status.lines() {
        let mut cols = line.split('\t');
        let Some(code) = cols.next() else { continue };
        // Renames report both old and new path; the new one is what matters.
        let path = match cols.clone().count() {
            0 => continue,
            1 => cols.next().unwrap(),
            _ => {
                let _old = cols.next();
                cols.next().unwrap()
            }
        };

        let deleted = code.starts_with('D');
        if opts.patterns.is_test(path) {
            if deleted {
                deleted_test_paths.push(path.to_string());
            } else {
                test_paths.push(path.to_string());
            }
        } else if !deleted {
            source_paths.push(path.to_string());
        } else {
            // A deleted source file is still part of the change the agent has
            // to reproduce, so it counts toward difficulty.
            source_paths.push(path.to_string());
        }
    }

    // Both halves are required. Without a test change there is nothing to
    // fail; without a source change there is nothing to fix.
    if test_paths.is_empty() || source_paths.is_empty() {
        return Ok(None);
    }

    // A subject that is blank gives the agent nothing to go on.
    if c.subject.is_empty() {
        return Ok(None);
    }

    let (lines_added, lines_removed) = numstat(repo, &c.sha, &opts.patterns)?;

    let split = if c.committed_at.as_str() < opts.cutoff.as_str() {
        Split::Train
    } else {
        Split::HeldOut
    };

    Ok(Some(Task {
        id: TaskId(format!("{}@{}", opts.corpus_name, &c.sha[..12.min(c.sha.len())])),
        corpus: opts.corpus_name.clone(),
        commit: c.sha.clone(),
        parent: parent.clone(),
        prompt: c.subject.clone(),
        difficulty: Difficulty {
            source_files_changed: source_paths.len() as u32,
            test_files_changed: (test_paths.len() + deleted_test_paths.len()) as u32,
            lines_added,
            lines_removed,
        },
        test_paths,
        deleted_test_paths,
        source_paths,
        oracle: opts.oracle.clone(),
        committed_at: c.committed_at.clone(),
        split,
        // Frontier tasks are authored, not mined; nothing here produces one.
        witness: None,
        // Git-mined tasks are coding tasks; reasoning tasks are imported, not
        // mined from history.
        kind: TaskKind::Coding,
    }))
}

/// Added and removed line counts across the commit's *source* files only.
///
/// Test churn is excluded: the agent is handed the tests, so their size is not
/// work it has to do, and counting them would inflate the difficulty of
/// commits that merely added a large fixture.
fn numstat(
    repo: &Path,
    sha: &str,
    patterns: &TestPatterns,
) -> Result<(u32, u32), CorpusError> {
    let out = git(
        repo,
        ["diff-tree", "--no-commit-id", "--numstat", "-r", "-M", sha],
    )?;
    let mut added = 0u32;
    let mut removed = 0u32;
    for line in out.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            continue;
        }
        let path = cols[cols.len() - 1];
        if patterns.is_test(path) {
            continue;
        }
        // Binary files report "-"; they contribute no line counts.
        added += cols[0].parse::<u32>().unwrap_or(0);
        removed += cols[1].parse::<u32>().unwrap_or(0);
    }
    Ok((added, removed))
}
