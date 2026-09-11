//! Run the Anti-Christ against a *deliberately weakened* guard, to mine
//! positive training signal.
//!
//!     cargo run -p samaritan-adversary --example train_target
//!
//! When the real guards hold everything — the healthy case, and what an
//! eight-round bout produced — the Deviant lands nothing and there is no
//! positive example to fine-tune on. This bout throws it at a foil: a target
//! with exactly one documented hole (an admission screen that does not
//! normalise path separators), so the backslash evasions the Deviant keeps
//! reaching for finally land. Every other attack class still meets the real
//! guard.
//!
//! Read the honest caveats in `samaritan_adversary::training` before trusting
//! the output: a landing here is a *foil* for a fine-tune, not a breach, and
//! the ledger it writes is a throwaway that must never be mixed into a
//! measured Solo/Critic/Adversarial run. Set LEDGER_DB to persist it, then run
//! the `export_dataset` example on that file.

use std::time::Duration;

use samaritan_adversary::training::{attempt_training, WeakGuard};
use samaritan_adversary::deviant::GenerativeDeviant;
use samaritan_adversary::{opening_book, summarise, Arena, ArenaConfig, Attacker, Target};
use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_dsl::MutationPolicy;
use samaritan_kernel::{Admission, AutonomyCeiling, ComputeBudget, Promotions};
use samaritan_ledger::{FixedClock, Ledger};

/// Cycles the opening book, so a model failure never stalls the bout.
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
    let sandbox = std::env::temp_dir().join("train-target-arena");
    let _ = std::fs::create_dir_all(&sandbox);
    Target {
        admission: Admission::default(),
        ceiling: AutonomyCeiling::closed(),
        promotions: Promotions::none(),
        budget: ComputeBudget::new(1_000_000),
        sandbox,
    }
}

fn main() {
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());
    let rounds: u64 = std::env::var("ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(66_600);

    let agent = Agent::new(AgentConfig {
        base_url: base_url.clone(),
        model: "samaritan-playout".into(),
        temperature: 1.0,
        max_tokens: 2048,
        constrain: Constrain::Grammar,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        seed: Some(seed),
        ..Default::default()
    });

    let weak = WeakGuard::AdmissionSkipsSeparatorNormalization;
    println!("server:    {base_url}");
    println!("adversary: the Anti-Christ (live model), fallback = opening book");
    println!("target:    WEAKENED foil ({weak:?}) — landings are training labels, not breaches\n");

    let mut deviant = GenerativeDeviant::new(agent, 1.0, seed, BookAttacker(0));
    let mut arena = Arena::new(ArenaConfig::default());
    let mut ledger = match std::env::var("LEDGER_DB") {
        Ok(path) => {
            let _ = std::fs::remove_file(&path);
            Ledger::open_with_clock(&path, Box::new(FixedClock("2026-09-11T00:00:00Z".into())))
                .unwrap()
        }
        Err(_) => Ledger::in_memory(Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap(),
    };
    let t = target();
    let policy = MutationPolicy::new();

    for _ in 0..rounds {
        let known = arena.breached();
        let attack = deviant.propose(&known, &policy);
        // The one difference from a measured bout: the verdict comes from the
        // weakened entry point. The arena bookkeeping is identical.
        let verdict = attempt_training(&attack, weak, &t);
        let round = arena.record(&attack, verdict.clone(), &mut ledger).unwrap();

        let source = if deviant.last_fell_back.is_some() { "book" } else { "model" };
        println!("── round {} [{source}] ──────────────────────────────", round.round);
        println!("   class:   {:?}", attack.class());
        println!("   intent:  {}", attack.note());
        match &verdict {
            samaritan_adversary::Verdict::Landed { evidence } => println!("   LANDED:  {evidence}"),
            samaritan_adversary::Verdict::Repelled { by } => println!("   repelled: {by}"),
            samaritan_adversary::Verdict::Inert { why } => println!("   inert:    {why}"),
        }
        println!(
            "   novelty: {:.2}   reward: {:.2}   containment(foil): {:.2}\n",
            round.novelty, round.reward.0, round.containment
        );
        if let Some(err) = &deviant.last_fell_back {
            println!("   (model call failed, fell back: {err})\n");
        }
    }

    let summary = summarise(&ledger).expect("summarise");
    println!(
        "landed {} / repelled {} / inert {} — mean reward {:.3}",
        summary.landed, summary.repelled, summary.inert, summary.mean_reward
    );
    if summary.has_positive_signal() {
        println!("positive signal present: export this ledger and it has something to teach.");
    } else {
        println!("still no landing — even the foil repelled everything; widen the hole or the book.");
    }
    ledger.verify().unwrap();
}
