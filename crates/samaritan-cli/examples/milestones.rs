//! Show a run's capability-milestone status.
//!
//!     LEDGER_DB=run.db cargo run -p samaritan-cli --example milestones
//!
//! Reads the evidence a ledger records — calibration, containment streak,
//! certificate-survived self-mods, the latest HLE score — and prints, per
//! capability, whether the bar is met and what is still missing. It reports
//! eligibility; it grants nothing. Respects NO_COLOR.

use samaritan_cli::render::{milestones, Style};
use samaritan_ledger::Ledger;

fn main() {
    let db = std::env::var("LEDGER_DB").unwrap_or_else(|_| {
        eprintln!("set LEDGER_DB to a ledger to inspect");
        std::process::exit(2);
    });
    let ledger = Ledger::open(&db).unwrap_or_else(|e| {
        eprintln!("could not open {db}: {e}");
        std::process::exit(1);
    });
    let ev = ledger.milestone_evidence().unwrap_or_else(|e| {
        eprintln!("could not read evidence: {e}");
        std::process::exit(1);
    });

    let style = if std::env::var_os("NO_COLOR").is_some() {
        Style::PLAIN
    } else {
        Style::COLOR
    };
    print!("{}", milestones(&ev, 72, style));
}
