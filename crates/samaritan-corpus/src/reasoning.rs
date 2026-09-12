//! Loading reasoning tasks from a JSONL file.
//!
//! Coding tasks are *mined* from a repository's history; reasoning tasks are
//! *imported* — a math item from NuminaMath, a science item from GPQA, an HLE
//! question held out for transfer. This reads them into a [`Corpus`] of
//! [`TaskKind::Reasoning`] tasks so the same episode/search/certificate
//! machinery scores them.
//!
//! The leakage discipline is the caller's to set and the type system's to
//! enforce: pass `trainable = false` for anything the search must never see
//! (HLE, GPQA), and [`crate::Manifest::check_disjoint`] will refuse a run that
//! trains on a repository it also measures transfer against. HLE ships no
//! training split, so an HLE corpus is *always* held-out.

use serde::Deserialize;

use crate::{AnswerKind, Corpus, CorpusError, Split, Task, TaskId};

/// One line of a reasoning JSONL. Extra fields are ignored, so a dataset with
/// more columns than this loads unchanged.
#[derive(Debug, Clone, Deserialize)]
struct ReasoningItem {
    /// Optional stable id; a positional one is assigned if absent.
    #[serde(default)]
    id: Option<String>,
    question: String,
    answer: String,
    #[serde(default = "default_answer_kind", alias = "answer_type")]
    answer_kind: AnswerKind,
    #[serde(default)]
    domain: String,
    /// Per-item split, if the dataset carries one; otherwise the loader's
    /// default is used.
    #[serde(default)]
    split: Option<Split>,
}

fn default_answer_kind() -> AnswerKind {
    AnswerKind::ExactMatch
}

/// Load reasoning tasks from a JSONL blob into a corpus.
///
/// `default_split` applies to any item that does not carry its own — so a
/// held-out HLE file loads with `Split::HeldOut` and a NuminaMath training file
/// with `Split::Train`, without either needing a per-line split. Blank lines are
/// skipped; a malformed line is an error naming its number, so a bad export
/// fails loudly rather than silently dropping questions.
pub fn load_reasoning(
    jsonl: &str,
    corpus_name: &str,
    trainable: bool,
    default_split: Split,
) -> Result<Corpus, CorpusError> {
    let mut tasks = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let item: ReasoningItem = serde_json::from_str(line).map_err(|e| {
            CorpusError::Invalid(format!("reasoning item on line {}: {e}", i + 1))
        })?;
        let id = item
            .id
            .unwrap_or_else(|| format!("{corpus_name}#{}", i + 1));
        let split = item.split.unwrap_or(default_split);
        tasks.push(Task::reasoning(
            TaskId(id),
            corpus_name,
            item.question,
            item.answer,
            item.answer_kind,
            item.domain,
            // Reasoning tasks have no commit date; the split is explicit above,
            // not derived from time, so this is a stable placeholder.
            "2026-01-01T00:00:00+00:00",
            split,
        ));
    }
    Ok(Corpus {
        name: corpus_name.to_string(),
        repo_path: String::new(),
        // No cutoff: a reasoning corpus splits by the explicit per-task split,
        // not by a commit date.
        cutoff: String::new(),
        trainable,
        tasks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_items_and_assigns_ids_and_split() {
        let jsonl = "\
{\"question\":\"6 times 7?\",\"answer\":\"42\",\"answer_kind\":\"exactMatch\",\"domain\":\"math\"}
{\"id\":\"q2\",\"question\":\"pick one\",\"answer\":\"C\",\"answer_type\":\"multipleChoice\",\"domain\":\"logic\",\"split\":\"held_out\"}
";
        let c = load_reasoning(jsonl, "sample", true, Split::Train).unwrap();
        assert_eq!(c.tasks.len(), 2);
        assert!(c.trainable);
        // Positional id when absent, explicit id honored.
        assert_eq!(c.tasks[0].id.0, "sample#1");
        assert_eq!(c.tasks[1].id.0, "q2");
        // Default split applied to the first, per-item split to the second.
        assert_eq!(c.tasks[0].split, Split::Train);
        assert_eq!(c.tasks[1].split, Split::HeldOut);
        // Graded as reasoning.
        assert_eq!(c.tasks[0].grade("the answer is 42"), Some(true));
        assert_eq!(c.tasks[0].domain(), "math");
    }

    #[test]
    fn a_held_out_corpus_trains_nothing() {
        // The HLE shape: not trainable, so the search can never see it even
        // though its items are Split::HeldOut.
        let jsonl = "{\"question\":\"hard\",\"answer\":\"x\",\"domain\":\"hle\"}\n";
        let c = load_reasoning(jsonl, "hle", false, Split::HeldOut).unwrap();
        assert!(!c.trainable);
        assert!(c.trainable_tasks().is_empty());
    }

    #[test]
    fn a_malformed_line_fails_loudly_with_its_number() {
        let jsonl = "{\"question\":\"ok\",\"answer\":\"a\"}\nnot json\n";
        match load_reasoning(jsonl, "x", true, Split::Train) {
            Err(CorpusError::Invalid(m)) => assert!(m.contains("line 2"), "{m}"),
            other => panic!("expected a line-2 error, got {other:?}"),
        }
    }
}
