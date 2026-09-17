//! Grading as a service, so a non-Rust trainer cannot drift from the oracle.
//!
//!     cargo run -q --release -p samaritan-corpus --example grade_batch
//!
//! Reads JSONL on stdin, writes JSONL on stdout, one line out per line in:
//!
//!     in:  {"given": "...", "answer": "42", "answer_kind": "exactMatch",
//!           "question": "...", "steps": [{"text": "...", "value": "28"}]}
//!     out: {"correct": true, "step_recall": 0.75}
//!
//! Why this exists: RL with verifiable rewards needs a reward signal, and the
//! trainer is Python. Reimplementing the grader there would create a second
//! oracle that silently disagrees with the one every eval in this project used —
//! and a reward function that disagrees with the measurement is how a model gets
//! optimised toward the wrong thing. The harness owns grading; Python asks it.
//!
//! `given` is the model's raw reply: the answer is lifted from it exactly the way
//! `run_reasoning_episode` does (drop the thinking block, prefer an `Answer:`
//! marker), so a rollout is scored by the same path a graded episode is.
//!
//! WHY `step_recall`. A binary correct/incorrect reward carries no gradient when
//! every rollout in a group agrees, and on this curriculum they usually do: the
//! base solves generated d3 at 39/40, and a 40-step GRPO run on it produced one
//! update. Eight rollouts that are all wrong are not all *equally* wrong, though
//! — one may have reached the fourth intermediate quantity before losing the
//! thread, another may never have started. `steps` is the generator's own
//! solution path, recorded at construction time, so that difference is
//! measurable without a reward model and without a judge.
//!
//! This is scored against the generator's route, which is a real bias: a model
//! taking a different valid path scores lower for it. That is why the trainer
//! weights it well below correctness — it separates rollouts that the binary
//! grader calls identical, and never outranks getting the answer right.
//!
//! Anchors are deliberately conservative, because the failure mode is a model
//! learning to sprinkle plausible numbers rather than to reason:
//!   * integers only — `"17 mod 20"` is a restatement, not a derived quantity;
//!   * at least two digits — single digits occur by chance in any long trace;
//!   * absent from the question — a given is not progress, and copying the
//!     question would otherwise score;
//!   * matched on digit boundaries — `28` does not match inside `128`, `28.5`
//!     or `-28`.
//! A problem with fewer than two surviving anchors yields `null` rather than a
//! number: no signal is honest, a constant is not.
//!
//! Errors are per line: a malformed line yields `{"correct": false, "error": ...}`
//! rather than killing the stream, because a trainer mid-rollout should degrade
//! to "no reward" rather than crash.

use std::io::{BufRead, Write};

use samaritan_corpus::{extract_answer, grade_answer, AnswerKind};

/// At least two digits, optional leading minus, nothing else. Anything with a
/// space, a letter or an operator in it is a restatement of the problem rather
/// than a quantity the solver had to derive.
fn is_anchor_literal(v: &str) -> bool {
    let digits = v.strip_prefix('-').unwrap_or(v);
    digits.len() >= 2 && digits.bytes().all(|b| b.is_ascii_digit())
}

/// Does `needle` occur in `hay` as a whole numeric token?
///
/// The preceding character may not be a digit, letter, `.`, `_` or `-`, so
/// `28` matches neither `128` nor `-28`; the following character may not be a
/// digit, letter, `.` or `_`, so it does not match inside `28.5` or `283` —
/// but `337-2` does contain `337`, which is why the two sides differ.
fn contains_token(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let (h, n) = (hay.as_bytes(), needle.as_bytes());
    if n.len() > h.len() {
        return false;
    }
    for i in 0..=(h.len() - n.len()) {
        if &h[i..i + n.len()] != n {
            continue;
        }
        let before_ok = i == 0 || {
            let c = h[i - 1];
            !(c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'-')
        };
        let j = i + n.len();
        let after_ok = j == h.len() || {
            let c = h[j];
            !(c.is_ascii_alphanumeric() || c == b'.' || c == b'_')
        };
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// The distinct derived quantities on the generator's path that a solver could
/// not have copied from the question.
fn anchors(question: &str, steps: &serde_json::Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let Some(list) = steps.as_array() else {
        return out;
    };
    for s in list {
        let v = s["value"].as_str().unwrap_or("").trim();
        if !is_anchor_literal(v) || contains_token(question, v) || out.iter().any(|o| o == v) {
            continue;
        }
        out.push(v.to_string());
    }
    out
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut n = 0u64;
    let mut correct = 0u64;
    let mut shaped = 0u64;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                let _ = writeln!(out, "{{\"correct\":false,\"error\":\"read: {e}\"}}");
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(out, "{{\"correct\":false,\"error\":\"parse: {e}\"}}");
                let _ = out.flush();
                continue;
            }
        };
        let given_raw = v["given"].as_str().unwrap_or("");
        let expected = v["answer"].as_str().unwrap_or("");
        let kind = match v["answer_kind"].as_str().unwrap_or("exactMatch") {
            "multipleChoice" => AnswerKind::MultipleChoice,
            _ => AnswerKind::ExactMatch,
        };
        // Same extraction the episode uses, so a rollout is scored exactly as a
        // graded answer would be — not on the raw trace.
        let (answer, confidence) = extract_answer(given_raw);
        let ok = !expected.is_empty() && grade_answer(expected, kind, &answer);
        n += 1;
        if ok {
            correct += 1;
        }

        // Recall is measured over the WHOLE reply, thinking block included: the
        // intermediate quantities live in the working, which `extract_answer`
        // discards by design.
        let found = anchors(v["question"].as_str().unwrap_or(""), &v["steps"]);
        let recall = if found.len() < 2 {
            serde_json::Value::Null
        } else {
            shaped += 1;
            let hit = found.iter().filter(|a| contains_token(given_raw, a)).count();
            serde_json::json!(hit as f64 / found.len() as f64)
        };

        let rec = serde_json::json!({
            "correct": ok,
            "extracted": answer,
            "confidence": confidence,
            "step_recall": recall,
        });
        let _ = writeln!(out, "{rec}");
        let _ = out.flush();
    }
    eprintln!("graded {n} item(s), {correct} correct, {shaped} with step anchors");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_match_respects_digit_boundaries() {
        assert!(contains_token("so x = 28 here", "28"));
        assert!(contains_token("28", "28"));
        assert!(contains_token("(28)", "28"));
        assert!(!contains_token("128", "28"), "must not match a suffix");
        assert!(!contains_token("283", "28"), "must not match a prefix");
        assert!(!contains_token("28.5", "28"), "must not match a decimal head");
        assert!(!contains_token("-28", "28"), "a negative is a different value");
        assert!(contains_token("337-2", "337"), "subtraction is still an occurrence");
    }

    #[test]
    fn anchors_drop_what_a_solver_never_had_to_derive() {
        let steps = serde_json::json!([
            {"text": "restate", "value": "17 mod 20"},   // not an integer
            {"text": "given",   "value": "20"},          // already in the question
            {"text": "trivial", "value": "1"},           // single digit
            {"text": "derived", "value": "337"},
            {"text": "again",   "value": "337"},         // duplicate
            {"text": "note",    "value": null},
        ]);
        let got = anchors("remainder 17 when divided by 20", &steps);
        assert_eq!(got, vec!["337".to_string()]);
    }

    #[test]
    fn recall_counts_distinct_anchors_reached() {
        let steps = serde_json::json!([
            {"text": "a", "value": "28"},
            {"text": "b", "value": "56"},
            {"text": "c", "value": "13"},
            {"text": "d", "value": "99"},
        ]);
        let found = anchors("nothing here", &steps);
        assert_eq!(found.len(), 4);
        let reply = "first 28, then 56, then I lost the thread";
        let hit = found.iter().filter(|a| contains_token(reply, a)).count();
        assert_eq!(hit, 2, "half the path reached");
    }
}
