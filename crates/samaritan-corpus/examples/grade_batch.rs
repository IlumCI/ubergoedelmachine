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

use samaritan_corpus::{extract_answer, grade_answer, step_recall, AnswerKind};

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

        let values: Vec<Option<&str>> = v["steps"]
            .as_array()
            .map(|l| l.iter().map(|s| s["value"].as_str()).collect())
            .unwrap_or_default();
        let recall = match step_recall(
            given_raw,
            v["question"].as_str().unwrap_or(""),
            values,
        ) {
            Some(r) => {
                shaped += 1;
                serde_json::json!(r)
            }
            None => serde_json::Value::Null,
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
    use samaritan_corpus::step_recall;

    /// The anchor rules themselves are tested in the library. This covers the
    /// wiring: JSON `steps` in, a number or null out.
    #[test]
    fn json_steps_reach_the_scorer() {
        let steps = serde_json::json!([
            {"text": "a", "value": "28"},
            {"text": "b", "value": "56"},
            {"text": "c", "value": "13"},
            {"text": "structural", "value": null},
        ]);
        let values: Vec<Option<&str>> =
            steps.as_array().unwrap().iter().map(|s| s["value"].as_str()).collect();
        let got = step_recall("first 28, then 56, then I lost it", "nothing here", values);
        assert_eq!(got, Some(2.0 / 3.0));
    }

    #[test]
    fn a_problem_with_no_steps_scores_nothing_rather_than_zero() {
        let none: Vec<Option<&str>> = Vec::new();
        assert_eq!(step_recall("anything", "a question", none), None);
    }
}
