//! Tests for the approval prompt.
//!
//! Rendering is pure, so the properties that matter are assertable without a
//! terminal. They are mostly about honesty: the prompt must show what will
//! actually happen, must not misreport it, and must not offer a shortcut the
//! rest of the system is careful to make expensive.

use samaritan_cli::{Style, preview};
use samaritan_dsl::{
    ActionKind, BlastRadius, Confidence, Prediction, ProposedAction, Reversibility,
};
use samaritan_kernel::{ApprovalRequest, Tier};

fn action(kind: ActionKind, rev: Reversibility, blast: BlastRadius, payload: serde_json::Value) -> ProposedAction {
    ProposedAction {
        kind,
        reversibility: rev,
        blast_radius: blast,
        intent: "do the thing".into(),
        payload,
    }
}

fn render_at(a: &ProposedAction, tier: Tier, confidence: f64, width: usize) -> String {
    let prediction = Prediction {
        outcome: "tests pass".into(),
        confidence: Confidence::new(confidence).unwrap(),
    };
    let req = ApprovalRequest {
        action: a,
        rationale: "the smallest change that satisfies the test".into(),
        prediction: &prediction,
        tier,
    };
    preview(&req, width, Style::PLAIN)
}

fn render(a: &ProposedAction, tier: Tier, confidence: f64) -> String {
    let prediction = Prediction {
        outcome: "tests pass".into(),
        confidence: Confidence::new(confidence).unwrap(),
    };
    let req = ApprovalRequest {
        action: a,
        rationale: "the smallest change that satisfies the test".into(),
        prediction: &prediction,
        tier,
    };
    preview(&req, 74, Style::PLAIN)
}

fn write_action(contents: &str) -> ProposedAction {
    action(
        ActionKind::Write,
        Reversibility::Snapshot,
        BlastRadius::Repo,
        serde_json::json!({"do": "write_file", "path": "src/lib.rs", "contents": contents}),
    )
}

// ------------------------------------------------------ showing the truth

#[test]
fn code_indentation_survives_rendering() {
    // The bug that shipped in the first draft: word wrapping ate leading
    // whitespace, so a reviewer approving a patch was shown bytes other than
    // the ones about to be written.
    let out = render(
        &write_action("fn main() {\n    let x = 1;\n        deeper();\n}\n"),
        Tier::Confirm,
        0.8,
    );
    assert!(out.contains("    let x = 1;"), "4-space indent lost:\n{out}");
    assert!(out.contains("        deeper();"), "8-space indent lost:\n{out}");
}

#[test]
fn a_blank_line_inside_a_patch_is_preserved() {
    let out = render(&write_action("a\n\nb\n"), Tier::Confirm, 0.8);
    let body: Vec<&str> = out
        .lines()
        .skip_while(|l| !l.contains('│'))
        .take_while(|l| l.contains('│'))
        .collect();
    assert!(
        body.len() >= 5,
        "expected title, gap, a, blank, b; got {body:#?}"
    );
}

#[test]
fn the_command_is_shown_as_it_will_be_run() {
    let a = action(
        ActionKind::Exec,
        Reversibility::Snapshot,
        BlastRadius::Machine,
        serde_json::json!({"do": "run", "program": "cargo", "args": ["test", "--quiet"], "timeout_secs": 300}),
    );
    let out = render(&a, Tier::Confirm, 0.8);
    assert!(out.contains("cargo test --quiet"), "got:\n{out}");
}

#[test]
fn danger_is_spelled_out_rather_than_named() {
    // "Repo" tells a human nothing. "in the real repository" tells them
    // whether to be worried, which is the only reason to interrupt them.
    let out = render(&write_action("x"), Tier::Confirm, 0.8);
    assert!(out.contains("in the real repository"), "got:\n{out}");
    assert!(out.contains("undone from the snapshot"), "got:\n{out}");
}

#[test]
fn a_long_file_is_truncated_with_the_count_shown() {
    let big: String = (0..80).map(|i| format!("line {i}\n")).collect();
    let out = render(&write_action(&big), Tier::Confirm, 0.8);
    assert!(out.contains("more lines"), "got:\n{out}");
    assert!(!out.contains("line 79"), "the whole file was dumped");
}

#[test]
fn an_unknown_payload_still_shows_something_reviewable() {
    // A shape the renderer does not recognise must not render as an empty
    // box: falling back to the raw JSON is ugly and honest.
    let a = action(
        ActionKind::Net,
        Reversibility::Costly,
        BlastRadius::External,
        serde_json::json!({"do": "telepathy", "target": "somewhere"}),
    );
    let out = render(&a, Tier::Confirm, 0.8);
    assert!(out.contains("telepathy"), "got:\n{out}");
}

// -------------------------------------------- the agent's own reasoning

#[test]
fn the_prediction_and_its_confidence_are_shown() {
    // The reviewer is judging two things: whether the action may run, and
    // whether the agent is right about what it will do. The second is
    // invisible if the prompt shows only the command.
    let out = render(&write_action("x"), Tier::Confirm, 0.88);
    assert!(out.contains("tests pass"), "got:\n{out}");
    assert!(out.contains("confidence 0.88"), "got:\n{out}");
    assert!(
        out.contains("the smallest change that satisfies the test"),
        "the rationale should be shown:\n{out}"
    );
}

// ------------------------------------------------------- what it refuses

#[test]
fn there_is_no_standing_permission_option() {
    // The deliberate departure from Claude Code. Autonomy here is earned by
    // a clean streak plus a certificate; a button would let a tired human
    // grant in one keystroke what the rest of the system makes expensive.
    let out = render(&write_action("x"), Tier::Confirm, 0.8);
    let lower = out.to_lowercase();
    for phrase in ["don't ask again", "always allow", "yes to all", "remember"] {
        // The explanatory footnote mentions the absence once; what must not
        // exist is a numbered option offering it.
        assert!(
            !out.contains(&format!("3. {phrase}")),
            "found a standing-permission option: {phrase}"
        );
    }
    assert!(lower.contains("autonomy is earned"), "got:\n{out}");
    assert!(out.contains("1. Yes"));
    assert!(out.contains("2. No"));
    assert!(!out.contains("3. "), "there are exactly two options:\n{out}");
}

#[test]
fn a_denial_is_not_presented_as_a_question() {
    let a = action(
        ActionKind::GitHistory,
        Reversibility::Irreversible,
        BlastRadius::External,
        serde_json::json!({"do": "run", "program": "git", "args": ["push", "--force"]}),
    );
    let out = render(&a, Tier::Deny, 0.7);
    assert!(!out.contains("Do you want to proceed?"), "got:\n{out}");
    assert!(!out.contains("1. Yes"), "a denial must offer no way through");
    assert!(out.contains("refused by the frozen rules"), "got:\n{out}");
}

// ------------------------------------------------------------- geometry

#[test]
fn the_box_stays_rectangular_at_every_width() {
    for width in [48, 60, 74, 100, 200] {
        let out = render_at(
            &write_action("a somewhat long line of source code here"),
            Tier::Confirm,
            0.8,
            width,
        );
        let widths: Vec<usize> = out
            .lines()
            .filter(|l| l.starts_with('│') || l.starts_with('╭') || l.starts_with('╰'))
            .map(|l| l.chars().count())
            .collect();
        assert!(!widths.is_empty());
        assert!(
            widths.iter().all(|w| *w == widths[0]),
            "ragged box at width {width}: {widths:?}"
        );
        // The renderer clamps to [48, 100]; within that the box must
        // actually track the width it was given, or the test above passes
        // for the wrong reason.
        assert_eq!(
            widths[0],
            width.clamp(48, 100),
            "box did not honour width {width}"
        );
    }
}

#[test]
fn a_very_long_unbroken_token_cannot_overflow_the_box() {
    let out = render(&write_action(&"x".repeat(500)), Tier::Confirm, 0.8);
    let widths: Vec<usize> = out
        .lines()
        .filter(|l| l.starts_with('│'))
        .map(|l| l.chars().count())
        .collect();
    assert!(
        widths.iter().all(|w| *w == widths[0]),
        "a long token broke the frame: {widths:?}"
    );
}

#[test]
fn colour_does_not_change_the_layout() {
    // Styling must not be counted as width, or every coloured line would be
    // padded short and the box would tear.
    let a = write_action("fn main() {}\n");
    let prediction = Prediction {
        outcome: "tests pass".into(),
        confidence: Confidence::new(0.8).unwrap(),
    };
    let req = ApprovalRequest {
        action: &a,
        rationale: "because".into(),
        prediction: &prediction,
        tier: Tier::Confirm,
    };
    let plain = preview(&req, 74, Style::PLAIN);
    let fancy = preview(&req, 74, Style::COLOR);

    let strip = |s: &str| {
        let mut out = String::new();
        let mut esc = false;
        for c in s.chars() {
            if esc {
                if c == 'm' {
                    esc = false;
                }
            } else if c == '\x1b' {
                esc = true;
            } else {
                out.push(c);
            }
        }
        out
    };
    assert_eq!(strip(&fancy), plain, "colour changed the rendered text");
}
