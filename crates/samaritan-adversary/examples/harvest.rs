//! Harvest model-authored landings across many seeds into one training set.
//!
//!     cargo run -p samaritan-adversary --example harvest
//!
//! One foil bout lands a handful of attacks; the QDoRA gate wants at least a
//! couple of dozen. This runs a bout per seed against a weakened foil, collects
//! every landing, and writes them merged as one JSONL the trainer reads.
//!
//! Env:
//!   SEED_BASE   first seed (default 1000)
//!   SEED_COUNT  how many consecutive seeds to run (default 8)
//!   ROUNDS      attacks per seed (default 8)
//!   WEAK        ceiling (default) | separator — which hole the foil exposes
//!   KNOWLEDGE   seed (default) | none — CWE priming for the adversary
//!   OUT         output JSONL (default harvest-dataset.jsonl)
//!   MIN_POSITIVE threshold the trainer wants (default 16), for the report only
//!
//! This is a *data-gathering* run, not a measured bout, and it differs from the
//! arena in one deliberate way: each round gets a fresh prompt, with no "you
//! already breached X, aim elsewhere" steering. That steering is right for a
//! real arms race — it drives the adversary toward new holes — but here it would
//! push the model off the very class the foil lets it land the moment it first
//! succeeds. A harvest wants volume in the winnable class, so it does not steer.
//! The landings are foils for training, never breaches of a real guard.

use std::time::Duration;

use samaritan_adversary::dataset::{export, to_jsonl, ExportConfig, TrainingRecord};
use samaritan_adversary::deviant::GenerativeDeviant;
use samaritan_adversary::training::{attempt_training, WeakGuard};
use samaritan_adversary::{opening_book, Arena, ArenaConfig, Attacker, Target};
use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_dsl::MutationPolicy;
use samaritan_kernel::{Admission, AutonomyCeiling, ComputeBudget, Promotions};
use samaritan_knowledge::KnowledgeBase;
use samaritan_ledger::{FixedClock, Ledger};

struct BookAttacker(usize);
impl Attacker for BookAttacker {
    fn propose(
        &mut self,
        _l: &[samaritan_ledger::ExploitClass],
        _p: &MutationPolicy,
    ) -> samaritan_adversary::Attack {
        let book = opening_book();
        let a = book[self.0 % book.len()].clone();
        self.0 += 1;
        a
    }
}

fn target() -> Target {
    let sandbox = std::env::temp_dir().join("harvest-arena");
    let _ = std::fs::create_dir_all(&sandbox);
    Target {
        admission: Admission::default(),
        ceiling: AutonomyCeiling::closed(),
        promotions: Promotions::none(),
        budget: ComputeBudget::new(1_000_000),
        sandbox,
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());
    let seed_base = env_u64("SEED_BASE", 1000);
    let seed_count = env_u64("SEED_COUNT", 8);
    let rounds = env_u64("ROUNDS", 8);
    let out = std::env::var("OUT").unwrap_or_else(|_| "harvest-dataset.jsonl".into());
    let min_positive = env_u64("MIN_POSITIVE", 16) as usize;
    let weak = match std::env::var("WEAK").as_deref() {
        Ok("separator") => WeakGuard::AdmissionSkipsSeparatorNormalization,
        _ => WeakGuard::CeilingOmitsUpperBound,
    };
    let use_knowledge = !matches!(std::env::var("KNOWLEDGE").as_deref(), Ok("none"));

    println!("server:  {base_url}");
    println!("foil:    {weak:?}");
    println!("seeds:   {seed_base}..{}", seed_base + seed_count);
    println!("rounds:  {rounds} per seed");
    println!("knowledge: {}\n", if use_knowledge { "seed CWE" } else { "none" });

    let target = target();
    let policy = MutationPolicy::new();
    let mut all_rows: Vec<TrainingRecord> = Vec::new();
    let mut model_landings = 0usize;
    let mut book_landings = 0usize;

    for i in 0..seed_count {
        let seed = seed_base + i;
        let agent = Agent::new(AgentConfig {
            base_url: base_url.clone(),
            model: "samaritan-playout".into(),
            temperature: 1.0,
            max_tokens: 640,
            repeat_penalty: 1.2,
            constrain: Constrain::Grammar,
            timeout: Duration::from_secs(900),
            max_retries: 1,
            seed: Some(seed),
            ..Default::default()
        });
        let mut deviant = GenerativeDeviant::new(agent, 1.0, seed, BookAttacker(0));
        if use_knowledge {
            deviant = deviant.with_knowledge(KnowledgeBase::seed());
        }
        let mut arena = Arena::new(ArenaConfig::default());
        let mut ledger =
            Ledger::in_memory(Box::new(FixedClock("2026-09-12T00:00:00Z".into()))).unwrap();

        let mut seed_landed = 0usize;
        for _ in 0..rounds {
            // No steering: a fresh prompt every round (see the module note).
            let attack = deviant.propose(&[], &policy);
            let from_book = deviant.last_fell_back.is_some();
            let verdict = attempt_training(&attack, weak, &target);
            let round = arena.record(&attack, verdict.clone(), &mut ledger).unwrap();
            if round.verdict.landed() {
                seed_landed += 1;
                if from_book { book_landings += 1 } else { model_landings += 1 }
            }
        }

        let rows = export(&ledger, &ExportConfig::landed_only()).unwrap();
        all_rows.extend(rows);
        println!("seed {seed}: {seed_landed} landed");
    }

    // Distinct by the attack the model must learn to emit, so near-identical
    // repeats do not masquerade as variety in the report. The trainer dedups
    // too; writing everything lets a reward-weighted objective still see counts.
    let mut distinct = std::collections::BTreeSet::new();
    for r in &all_rows {
        distinct.insert(r.completion.clone());
    }

    std::fs::write(&out, to_jsonl(&all_rows)).expect("write jsonl");

    println!("\n── harvest ─────────────────────────────");
    println!("landed total:   {} (model {model_landings}, book {book_landings})", all_rows.len());
    println!("distinct attacks: {}", distinct.len());
    println!("wrote {} rows to {out}", all_rows.len());
    if distinct.len() >= min_positive {
        println!(
            "\ndistinct landings ({}) clear --min-positive ({min_positive}). Train:\n  \
             python training/qdora_deviant.py {out}",
            distinct.len()
        );
    } else {
        println!(
            "\ndistinct landings ({}) below --min-positive ({min_positive}). Run more seeds \
             (raise SEED_COUNT) or rounds, then retrain — or pass --allow-small to the trainer \
             with eyes open.",
            distinct.len()
        );
    }
}
