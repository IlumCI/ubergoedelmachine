//! Drawing the approval prompt.
//!
//! Rendering is a pure function from a request to a string. That is not
//! fastidiousness: an approval prompt is the one piece of UI where being
//! wrong is a safety problem rather than an aesthetic one, and a renderer
//! that needs a terminal to exercise is a renderer nobody writes tests for.
//!
//! The layout follows Claude Code's approval: an inline bordered block in the
//! scrollback rather than a full-screen takeover, the operation shown
//! verbatim, then a short numbered choice. Taking over the screen would erase
//! the context the human is being asked to judge.
//!
//! One deliberate departure. Claude Code offers "yes, and don't ask again".
//! Samaritan does not, and the prompt says so. Standing permission here is
//! not a UI affordance — it is earned by a streak of clean approvals, a
//! calibration score good enough to deserve it, and a certificate, none of
//! which a tired human at 2am can shortcut. Offering the button would make
//! the rest of that machinery decorative.

use samaritan_dsl::{ActionKind, BlastRadius, ProposedAction, Reversibility};
use samaritan_kernel::{ApprovalRequest, Tier};

/// ANSI styling, switchable off for tests and dumb terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Style {
    pub color: bool,
}

impl Style {
    pub const PLAIN: Style = Style { color: false };
    pub const COLOR: Style = Style { color: true };

    fn wrap(self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    fn dim(self, s: &str) -> String {
        self.wrap("2", s)
    }
    fn bold(self, s: &str) -> String {
        self.wrap("1", s)
    }
    fn yellow(self, s: &str) -> String {
        self.wrap("33", s)
    }
    fn red(self, s: &str) -> String {
        self.wrap("31", s)
    }
    fn cyan(self, s: &str) -> String {
        self.wrap("36", s)
    }
}

/// Printable width, ignoring ANSI escapes.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            if c == 'm' {
                in_esc = false;
            }
        } else if c == '\x1b' {
            in_esc = true;
        } else {
            n += 1;
        }
    }
    n
}

/// Break a line at exactly `width` characters, preserving every byte.
///
/// Used for anything the human is being asked to approve. Word wrapping is
/// right for prose and wrong for code: `split_whitespace` silently eats
/// leading indentation, so a reviewer approving a patch would be shown
/// something other than the bytes that get written. A prompt that misreports
/// the thing it is asking about is worse than no prompt.
fn hard_wrap(s: &str, width: usize) -> Vec<String> {
    if s.is_empty() {
        return vec![String::new()];
    }
    let chars: Vec<char> = s.chars().collect();
    chars
        .chunks(width.max(1))
        .map(|c| c.iter().collect())
        .collect()
}

/// Wrap on word boundaries, breaking a word only when it cannot fit at all.
///
/// Prose only — see [`hard_wrap`] for anything quoted back to the reviewer.
fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in s.split('\n') {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if line.is_empty() {
                line.push_str(word);
            } else if line.chars().count() + 1 + word.chars().count() <= width {
                line.push(' ');
                line.push_str(word);
            } else {
                out.push(std::mem::take(&mut line));
                line.push_str(word);
            }
            while line.chars().count() > width {
                let head: String = line.chars().take(width).collect();
                let tail: String = line.chars().skip(width).collect();
                out.push(head);
                line = tail;
            }
        }
        out.push(line);
    }
    out
}

fn human_kind(k: ActionKind) -> &'static str {
    match k {
        ActionKind::Read => "Read",
        ActionKind::Write => "Write",
        ActionKind::Exec => "Run",
        ActionKind::Net => "Network",
        ActionKind::GitHistory => "Rewrite history",
    }
}

fn human_rev(r: Reversibility) -> &'static str {
    match r {
        Reversibility::Trivial => "trivially undone",
        Reversibility::Snapshot => "undone from the snapshot",
        Reversibility::Costly => "costly to undo",
        Reversibility::Irreversible => "cannot be undone",
    }
}

/// What the blast radius means in a sentence, because the enum name alone
/// does not tell a human whether to be worried.
fn human_blast(b: BlastRadius) -> &'static str {
    match b {
        BlastRadius::Episode => "inside the episode sandbox",
        BlastRadius::Repo => "in the real repository",
        BlastRadius::Machine => "anywhere on this machine",
        BlastRadius::External => "outside this machine",
    }
}

/// Title and body for the operation, pulled out of the payload.
///
/// Shows what will actually happen rather than the JSON. A human approving a
/// write should see the bytes, not a serialised struct — reviewing the thing
/// itself is the entire point of the pause.
fn operation(action: &ProposedAction) -> (String, Vec<String>) {
    let p = &action.payload;
    let get = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();

    match p.get("do").and_then(|v| v.as_str()) {
        Some("write_file") => {
            let body = p
                .get("contents")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .lines()
                .map(str::to_string)
                .collect();
            (format!("Write  {}", get("path")), body)
        }
        Some("read_file") => (format!("Read  {}", get("path")), vec![]),
        Some("list_dir") => (format!("List  {}", get("path")), vec![]),
        Some("delete_file") => (format!("Delete  {}", get("path")), vec![]),
        Some("run") => {
            let args = p
                .get("args")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            let cmd = format!("{} {}", get("program"), args);
            ("Run".to_string(), vec![cmd.trim().to_string()])
        }
        _ => (
            human_kind(action.kind).to_string(),
            serde_json::to_string_pretty(p)
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect(),
        ),
    }
}

/// How many body lines to show before collapsing.
const MAX_BODY: usize = 20;

/// Render the whole prompt.
///
/// `selected` is the highlighted option: 0 = allow, 1 = refuse.
pub fn approval(req: &ApprovalRequest, width: usize, st: Style, selected: usize) -> String {
    let width = width.clamp(48, 100);
    let inner = width - 4;
    let mut out = String::new();

    let (title, body) = operation(req.action);

    // --- the operation, boxed -------------------------------------------
    out.push_str(&st.dim(&format!("╭{}╮\n", "─".repeat(width - 2))));

    let mut boxline = |s: &str| {
        let pad = inner.saturating_sub(visible_len(s));
        out.push_str(&format!(
            "{} {}{} {}\n",
            st.dim("│"),
            s,
            " ".repeat(pad),
            st.dim("│")
        ));
    };

    boxline(&st.bold(&title));

    if !body.is_empty() {
        boxline("");
        let shown = body.len().min(MAX_BODY);
        for l in body.iter().take(shown) {
            for w in hard_wrap(l, inner) {
                boxline(&st.cyan(&w));
            }
        }
        if body.len() > shown {
            boxline(&st.dim(&format!("… {} more lines", body.len() - shown)));
        }
    }

    out.push_str(&st.dim(&format!("╰{}╯\n", "─".repeat(width - 2))));
    out.push('\n');

    // --- why it is being asked ------------------------------------------
    let tierword = match req.tier {
        Tier::Deny => st.red("denied"),
        Tier::Confirm => st.yellow("needs approval"),
        Tier::Auto => st.dim("automatic"),
    };
    let effect = format!(
        "{} · {} · {}",
        human_kind(req.action.kind),
        human_rev(req.action.reversibility),
        human_blast(req.action.blast_radius),
    );
    for l in wrap(&effect, width - 2) {
        out.push_str(&format!("  {}\n", st.dim(&l)));
    }
    out.push_str(&format!("  {tierword}\n\n"));

    // A denial is not a question. The router never calls the gate for one,
    // but a prompt that offered to override the frozen rules would be wrong
    // on its face even as a preview, and the rendering is the part a human
    // would believe.
    if req.tier == Tier::Deny {
        out.push_str(&format!(
            "  {}\n",
            st.dim("refused by the frozen rules; nothing is being asked")
        ));
        return out;
    }

    // --- the agent's own reasoning --------------------------------------
    // Shown because the question is not only "may this run" but "is the
    // agent right about what it will do". A confident wrong prediction is
    // the signal a reviewer most wants, and it is invisible if the prompt
    // shows only the command.
    let label = |k: &str| st.dim(&format!("{k:<8}"));
    for (i, l) in wrap(req.rationale, width - 11).into_iter().enumerate() {
        out.push_str(&format!(
            "  {}{}\n",
            if i == 0 { label("why") } else { label("") },
            l
        ));
    }
    let conf = req.prediction.confidence.get();
    let conf_s = format!("confidence {conf:.2}");
    let conf_styled = if conf >= 0.9 {
        st.yellow(&conf_s)
    } else {
        st.dim(&conf_s)
    };
    for (i, l) in wrap(&req.prediction.outcome, width - 11)
        .into_iter()
        .enumerate()
    {
        out.push_str(&format!(
            "  {}{}\n",
            if i == 0 { label("expects") } else { label("") },
            l
        ));
    }
    out.push_str(&format!("  {}{}\n\n", label(""), conf_styled));

    // --- the choice ------------------------------------------------------
    out.push_str("  Do you want to proceed?\n");
    for (i, text) in ["Yes", "No, refuse this action"].iter().enumerate() {
        let marker = if i == selected { "❯" } else { " " };
        let line = format!("{} {}. {}", marker, i + 1, text);
        out.push_str(&format!(
            "  {}\n",
            if i == selected {
                st.cyan(&line)
            } else {
                line
            }
        ));
    }
    out.push('\n');
    out.push_str(&format!(
        "  {}\n",
        st.dim("there is no \"don't ask again\" — autonomy is earned by a clean")
    ));
    out.push_str(&format!(
        "  {}\n",
        st.dim("streak and a certificate, not granted at the prompt")
    ));

    out
}
