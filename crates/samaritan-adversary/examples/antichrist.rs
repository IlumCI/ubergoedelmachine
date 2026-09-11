//! Watch the Anti-Christ propose a real attack, and see whether it lands.
//!
//!     cargo run -p samaritan-adversary --example antichrist
//!
//! Expects a local model server (scripts\serve.ps1). Override with
//! SAMARITAN_URL. This runs a handful of arena rounds driven by the live
//! Deviant model against the real guards, and prints what it tried and how it
//! was answered — the arms race, in earnest, for one short bout.

use std::time::Duration;

use samaritan_adversary::deviant::GenerativeDeviant;
use samaritan_adversary::{opening_book, Arena, ArenaConfig, Attacker, SeededAttacker, Target};
use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_dsl::MutationPolicy;
use samaritan_kernel::{Admission, AutonomyCeiling, ComputeBudget, Promotions};
use samaritan_ledger::{FixedClock, Ledger};

/// A fallback attacker that cycles the opening book, so a model failure never
/// stalls the bout.
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
    let sandbox = std::env::temp_dir().join("antichrist-arena");
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
    let rounds: u64 = std::env::var("ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    let agent = Agent::new(AgentConfig {
        base_url: base_url.clone(),
        model: "samaritan-playout".into(),
        temperature: 1.0, // the Anti-Christ is meant to be creative
        max_tokens: 2048,   // the Anti-Christ writes elaborate payloads; give it room
        constrain: Constrain::Grammar,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        seed: Some(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(66_600)),
        ..Default::default()
    });

    println!("server:    {base_url}");
    println!("adversary: the Anti-Christ (live model), fallback = opening book\n");

    let mut deviant = GenerativeDeviant::new(agent, 1.0, std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(66_600), BookAttacker(0));
    let mut arena = Arena::new(ArenaConfig::default());
    // Persist the ledger when asked, so the bout can be exported to a training
    // set afterward; otherwise keep it in memory.
    let mut ledger = match std::env::var("LEDGER_DB") {
        Ok(path) => {
            let _ = std::fs::remove_file(&path);
            Ledger::open_with_clock(&path, Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap()
        }
        Err(_) => Ledger::in_memory(Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap(),
    };
    let t = target();
    let policy = MutationPolicy::new();
    let _ = SeededAttacker::new(BookAttacker(0)); // keep the import honest

    for _ in 0..rounds {
        let known = arena.breached();
        let attack = deviant.propose(&known, &policy);
        let verdict = samaritan_adversary::attempt(&attack, &t);
        let round = arena.record(&attack, verdict.clone(), &mut ledger).unwrap();

        let source = if deviant.last_fell_back.is_some() {
            "book"
        } else {
            "model"
        };
        println!("── round {} [{source}] ──────────────────────────────", round.round);
        println!("   class:   {:?}", attack.class());
        println!("   intent:  {}", attack.note());
        match &verdict {
            samaritan_adversary::Verdict::Landed { evidence } => {
                println!("   LANDED:  {evidence}");
            }
            samaritan_adversary::Verdict::Repelled { by } => println!("   repelled: {by}"),
            samaritan_adversary::Verdict::Inert { why } => println!("   inert:    {why}"),
        }
        println!("   novelty: {:.2}   reward: {:.2}   containment: {:.2}\n",
            round.novelty, round.reward.0, round.containment);
        if let Some(err) = &deviant.last_fell_back {
            println!("   (model call failed, fell back: {err})\n");
        }
    }

    println!("final containment index: {:.2}", arena.containment());
    if arena.adversary_is_stale() {
        println!("the Anti-Christ has gone stale — every attack repelled, time for a harder objective");
    }
    ledger.verify().unwrap();
}
