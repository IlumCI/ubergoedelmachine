//! Export a run's bouts into a QDoRA training set.
//!
//!     cargo run -p samaritan-adversary --example export_dataset -- <ledger.db> [out.jsonl]
//!
//! Reads the attack attempts a run recorded, prints the class balance, and
//! writes reward-labelled JSONL. Reports plainly whether the set has any
//! positive signal — an all-repel run has nothing a fine-tune can learn from
//! except silence, and it is better to say so here than to discover it after
//! a GPU-afternoon.

use samaritan_adversary::dataset::{export, summarise, to_jsonl, ExportConfig};
use samaritan_ledger::Ledger;

fn main() {
    let mut args = std::env::args().skip(1);
    let db = args.next().unwrap_or_else(|| {
        eprintln!("usage: export_dataset <ledger.db> [out.jsonl]");
        std::process::exit(2);
    });
    let out = args.next().unwrap_or_else(|| "deviant-dataset.jsonl".to_string());
    let positives_only = std::env::var("POSITIVES_ONLY").is_ok();

    let ledger = match Ledger::open(&db) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("could not open ledger {db}: {e}");
            std::process::exit(1);
        }
    };

    let summary = summarise(&ledger).expect("summarise");
    println!("attempts: {}", summary.total);
    println!("  landed:   {}", summary.landed);
    println!("  repelled: {}", summary.repelled);
    println!("  inert:    {}", summary.inert);
    println!("  mean reward: {:.3}", summary.mean_reward);

    if !summary.has_positive_signal() {
        println!(
            "\nNo attempt landed. This set has no positive signal — a reward-weighted \
             fine-tune on it would learn to attempt nothing. Run more bouts, or tighten \
             the Deviant, before training."
        );
    }

    let cfg = if positives_only {
        ExportConfig::landed_only()
    } else {
        ExportConfig::default()
    };
    let rows = export(&ledger, &cfg).expect("export");
    std::fs::write(&out, to_jsonl(&rows)).expect("write jsonl");
    println!("\nwrote {} rows to {out}{}", rows.len(),
        if positives_only { " (positives only)" } else { "" });
}
