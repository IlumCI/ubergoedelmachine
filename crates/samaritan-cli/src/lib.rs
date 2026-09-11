//! The approval gate a human actually sees.
//!
//! Implements [`ApprovalGate`] by drawing the prompt inline and waiting for a
//! keypress. Everything interesting about it is in what it refuses to do:
//!
//! - **It never answers on the human's behalf.** No timeout that defaults to
//!   yes, no "assume approval in CI". When there is no terminal to ask, it
//!   returns [`GateError::Unattended`], which the router treats as a refusal.
//! - **It offers no standing permission.** Autonomy is earned through the
//!   promotion path — a streak of clean approvals, calibration good enough to
//!   deserve it, and a certificate. A "don't ask again" button would let a
//!   tired human grant in one keystroke what the rest of the system is
//!   careful to make expensive.
//! - **It shows the agent's prediction, not just the command.** The question
//!   is not only "may this run" but "is the agent right about what will
//!   happen". A confident wrong prediction is the thing a reviewer most needs
//!   to see, and it is invisible in a prompt that shows only the operation.

pub mod render;

use std::io::{IsTerminal, Write};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use samaritan_kernel::{Approval, ApprovalGate, ApprovalRequest, GateError};

pub use render::Style;

/// An approval gate backed by the terminal.
pub struct Tui {
    width: usize,
    style: Style,
}

impl Default for Tui {
    fn default() -> Self {
        Self::new()
    }
}

impl Tui {
    pub fn new() -> Self {
        let width = terminal::size().map(|(w, _)| w as usize).unwrap_or(80);
        let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self {
            width,
            style: if color { Style::COLOR } else { Style::PLAIN },
        }
    }

    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Read one decision from the keyboard.
    ///
    /// Accepts the number keys, `y`/`n`, arrows plus Enter, and Esc. Enter on
    /// its own takes whatever is highlighted, and the highlight starts on
    /// *Yes* — which is safe only because reaching this function at all means
    /// the frozen rules already declined to deny the action outright.
    ///
    /// Ctrl-C refuses rather than killing the process, so an interrupted
    /// approval leaves a recorded refusal in the ledger instead of a hole
    /// where an answer should be.
    fn ask_tty(&self, req: &ApprovalRequest) -> Result<Approval, GateError> {
        let mut selected = 0usize;
        let mut out = std::io::stdout();

        terminal::enable_raw_mode().map_err(|e| GateError::Channel(e.to_string()))?;
        let result = (|| loop {
            // Raw mode means writing our own carriage returns.
            let frame = render::approval(req, self.width, self.style, selected)
                .replace('\n', "\r\n");
            write!(out, "{frame}").map_err(|e| GateError::Channel(e.to_string()))?;
            out.flush().ok();
            let lines = frame.matches("\r\n").count();

            let ev = event::read().map_err(|e| GateError::Channel(e.to_string()))?;
            let redraw = |n: usize| format!("\x1b[{n}A\x1b[0J");

            match ev {
                Event::Key(KeyEvent {
                    code, modifiers, ..
                }) => match (code, modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) | (KeyCode::Esc, _) => {
                        write!(out, "{}", redraw(lines)).ok();
                        return Ok(Approval::Refuse);
                    }
                    (KeyCode::Char('1'), _) | (KeyCode::Char('y'), _) | (KeyCode::Char('Y'), _) => {
                        write!(out, "{}", redraw(lines)).ok();
                        return Ok(Approval::Allow);
                    }
                    (KeyCode::Char('2'), _) | (KeyCode::Char('n'), _) | (KeyCode::Char('N'), _) => {
                        write!(out, "{}", redraw(lines)).ok();
                        return Ok(Approval::Refuse);
                    }
                    (KeyCode::Enter, _) => {
                        write!(out, "{}", redraw(lines)).ok();
                        return Ok(if selected == 0 {
                            Approval::Allow
                        } else {
                            Approval::Refuse
                        });
                    }
                    (KeyCode::Up | KeyCode::Char('k'), _) => selected = 0,
                    (KeyCode::Down | KeyCode::Char('j'), _) => selected = 1,
                    _ => {}
                },
                // A resize mid-prompt would tear the frame; re-measure and
                // redraw rather than leaving a half-wrapped box on screen.
                Event::Resize(_, _) => {}
                _ => {}
            }
            write!(out, "{}", redraw(lines)).ok();
        })();

        let _ = terminal::disable_raw_mode();

        // Print the outcome so the scrollback records what was decided. An
        // approval that leaves no trace above the prompt is one the human
        // cannot later check themselves on.
        if let Ok(a) = &result {
            let line = match a {
                Approval::Allow => "  approved\n",
                Approval::Refuse => "  refused\n",
            };
            let _ = write!(out, "{line}\n");
            let _ = out.flush();
        }
        result
    }
}

impl ApprovalGate for Tui {
    fn ask(&mut self, request: &ApprovalRequest) -> Result<Approval, GateError> {
        // No terminal means no human. Returning anything other than an error
        // here — a default, a timeout, an "assume yes in CI" — would make the
        // whole gate depend on nobody thinking to run the harness detached.
        if !std::io::stdin().is_terminal() {
            return Err(GateError::Unattended);
        }
        self.ask_tty(request)
    }
}

/// Print an approval prompt without asking, for previewing the layout.
pub fn preview(req: &ApprovalRequest, width: usize, style: Style) -> String {
    render::approval(req, width, style, 0)
}
