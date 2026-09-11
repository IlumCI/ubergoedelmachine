//! Render the approval prompt for every action shape, without asking.
//!
//!     cargo run -p samaritan-cli --example preview
//!
//! A prompt is the one screen where being wrong is a safety problem, so it
//! wants looking at rather than only testing.

use samaritan_cli::{Style, preview};
use samaritan_dsl::{
    ActionKind, BlastRadius, Confidence, Prediction, ProposedAction, Reversibility,
};
use samaritan_kernel::{ApprovalRequest, Tier};

fn main() {
    let color = std::env::var_os("NO_COLOR").is_none();
    let style = if color { Style::COLOR } else { Style::PLAIN };
    let width = 74;

    let cases: Vec<(&str, ProposedAction, &str, &str, f64)> = vec![
        (
            "a write to the real repository",
            ProposedAction {
                kind: ActionKind::Write,
                reversibility: Reversibility::Snapshot,
                blast_radius: BlastRadius::Repo,
                intent: "fix the addition operator".into(),
                payload: serde_json::json!({
                    "do": "write_file",
                    "path": "src/lib.rs",
                    "contents": "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
                }),
            },
            "The test expects 4 from add(2,2) and the operator is subtraction. \
             Changing the operator is the smallest edit that satisfies it.",
            "tests/add.rs passes and no other test changes",
            0.88,
        ),
        (
            "a command, with an overconfident prediction",
            ProposedAction {
                kind: ActionKind::Exec,
                reversibility: Reversibility::Snapshot,
                blast_radius: BlastRadius::Machine,
                intent: "run the test suite".into(),
                payload: serde_json::json!({
                    "do": "run", "program": "cargo", "args": ["test", "--quiet"],
                    "timeout_secs": 300
                }),
            },
            "The edit is in place; running the suite settles whether it worked.",
            "the suite exits zero",
            0.95,
        ),
        (
            "something irreversible, shown for contrast",
            ProposedAction {
                kind: ActionKind::GitHistory,
                reversibility: Reversibility::Irreversible,
                blast_radius: BlastRadius::External,
                intent: "force-push the rewritten branch".into(),
                payload: serde_json::json!({
                    "do": "run", "program": "git",
                    "args": ["push", "--force", "origin", "main"], "timeout_secs": 60
                }),
            },
            "History is untidy and a rewrite would read better.",
            "the remote branch matches local",
            0.7,
        ),
    ];

    for (label, action, rationale, outcome, confidence) in cases {
        let prediction = Prediction {
            outcome: outcome.into(),
            confidence: Confidence::new(confidence).unwrap(),
        };
        // The third case is what the frozen rules would actually return for
        // an irreversible external action; it is drawn to show that a denial
        // is never presented as a question.
        let tier = if action.reversibility == Reversibility::Irreversible {
            Tier::Deny
        } else {
            Tier::Confirm
        };
        let req = ApprovalRequest {
            action: &action,
            rationale,
            prediction: &prediction,
            tier,
        };
        println!("\n\x1b[2m── {label} {}\x1b[0m\n", "─".repeat(60 - label.len()));
        print!("{}", preview(&req, width, style));
    }
    println!();
}
