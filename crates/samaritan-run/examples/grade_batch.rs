//! Grading as a service, so a non-Rust trainer cannot drift from the oracle.
//!
//!     cargo run -q -p samaritan-run --example grade_batch
//!
//! Reads JSONL on stdin, writes JSONL on stdout, one line out per line in:
//!
//!     in:  {"given": "...", "answer": "42", "answer_kind": "exactMatch"}
//!     out: {"correct": true}
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
//! Errors are per line: a malformed line yields `{"correct": false, "error": ...}`
//! rather than killing the stream, because a trainer mid-rollout should degrade
//! to "no reward" rather than crash.

use std::io::{BufRead, Write};

use samaritan_corpus::{grade_answer, AnswerKind};
use samaritan_episode::extract_answer;

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut n = 0u64;
    let mut correct = 0u64;

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
        let rec = serde_json::json!({
            "correct": ok,
            "extracted": answer,
            "confidence": confidence,
        });
        let _ = writeln!(out, "{rec}");
        let _ = out.flush();
    }
    eprintln!("graded {n} item(s), {correct} correct");
}
