//! The arms race: rounds of Deviant against Warden, and the measurements that
//! tell mutual improvement from mutual rot.
//!
//! One round is: the Deviant proposes an attack, it is thrown at the real
//! defences, and the verdict is recorded. A landed attack becomes a violation
//! in the ledger and a known exploit class the Warden must henceforth
//! withstand. The containment index is recomputed and written down.
//!
//! Two things this loop is built to make visible, because the coevolution
//! literature says both are the *default* outcome rather than the edge case:
//!
//! - **Farming.** A Deviant that re-lands one known trick forever drives the
//!   containment index nowhere while looking busy. Novelty scoring makes that
//!   attack worth almost nothing, so the incentive points at new holes.
//! - **Mutual rot.** Both sides can decay together while the *relative*
//!   containment index sits flat and reports health. The index alone cannot
//!   see this; the yardstick in `samaritan-ledger::diagnose` can, and the
//!   arena writes both every round so the pair is always available.
//!
//! The arena never runs the Deviant's code on the host. Attacks are data
//! evaluated against the guards; when an attack *does* need a process — a
//! real exploit attempt rather than a crafted input — that process is the
//! executor's job and runs under `Confinement::Container`, which is why the
//! design says the Deviant runs contained or does not run.

use std::collections::BTreeSet;

use samaritan_ledger::{containment_index, Actor, Event, ExploitClass, Ledger, LedgerError};
use serde::{Deserialize, Serialize};

use crate::attack::{attempt, landing_violation, Target, Verdict};
use crate::{AttackReward, Attacker};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    /// Rounds the Warden must repel everything before the Deviant's
    /// capability is bumped — a sign the adversary has gone stale and needs
    /// a harder objective, not that the system is safe.
    pub warden_hold_threshold: u32,
    /// Rounds of Deviant dominance before a Warden-regression alert — the
    /// adversary is winning too easily, which usually means the defence got
    /// worse rather than the attacks got better.
    pub deviant_runaway_threshold: u32,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            warden_hold_threshold: 5,
            deviant_runaway_threshold: 3,
        }
    }
}

/// The result of one round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Round {
    pub round: u64,
    pub verdict: Verdict,
    pub class: ExploitClass,
    pub reward: AttackReward,
    /// Novelty of the attack against the archive, `[0, 1]`.
    pub novelty: f64,
    /// Containment after this round: fraction of the whole attack surface the
    /// Warden still withstands.
    pub containment: f64,
    /// Whether this round newly breached a class.
    pub newly_breached: bool,
}

/// One coevolution run.
pub struct Arena {
    cfg: ArenaConfig,
    /// Classes ever breached, and the fingerprints ever *landed*, kept
    /// separately: the first drives the containment index, the second drives
    /// novelty. A class can be breached while a specific trick is novel.
    breached: BTreeSet<ExploitClass>,
    landed_fingerprints: BTreeSet<String>,
    round: u64,
    warden_hold_streak: u32,
    deviant_win_streak: u32,
}

impl Arena {
    pub fn new(cfg: ArenaConfig) -> Self {
        Self {
            cfg,
            breached: BTreeSet::new(),
            landed_fingerprints: BTreeSet::new(),
            round: 0,
            warden_hold_streak: 0,
            deviant_win_streak: 0,
        }
    }

    pub fn containment(&self) -> f64 {
        containment_index(&self.breached.iter().copied().collect::<Vec<_>>())
    }

    pub fn breached(&self) -> Vec<ExploitClass> {
        self.breached.iter().copied().collect()
    }

    /// How novel an attack is against everything landed so far.
    ///
    /// 1.0 for a fingerprint never landed, tapering as the same trick recurs.
    /// Deliberately keyed on *landed* tricks, not attempted ones: repelled
    /// attempts teach the Deviant nothing it can farm, so they should not
    /// depress the novelty of trying again.
    fn novelty(&self, fingerprint: &str) -> f64 {
        if self.landed_fingerprints.contains(fingerprint) {
            0.0
        } else {
            1.0
        }
    }

    /// Run one round against the real defences.
    pub fn step(
        &mut self,
        attacker: &mut dyn Attacker,
        target: &Target,
        policy: &samaritan_dsl::MutationPolicy,
        ledger: &mut Ledger,
    ) -> Result<Round, LedgerError> {
        self.round += 1;
        let known: Vec<ExploitClass> = self.breached.iter().copied().collect();
        let attack = attacker.propose(&known, policy);
        let fingerprint = attack.fingerprint();
        let novelty = self.novelty(&fingerprint);

        let verdict = attempt(&attack, target);
        let class = attack.class();
        let reward = AttackReward::score(&verdict, novelty);

        ledger.append(
            Actor::Deviant,
            &Event::Narration {
                text: format!(
                    "round {}: {} — {}",
                    self.round,
                    attack.note(),
                    match &verdict {
                        Verdict::Landed { evidence } => format!("LANDED: {evidence}"),
                        Verdict::Repelled { by } => format!("repelled ({by})"),
                        Verdict::Inert { why } => format!("inert ({why})"),
                    }
                ),
            },
        )?;

        let newly_breached = match &verdict {
            Verdict::Landed { evidence } => {
                let first_time = self.breached.insert(class);
                self.landed_fingerprints.insert(fingerprint);
                ledger.append(
                    Actor::Deviant,
                    &Event::ExploitLanded {
                        round: self.round,
                        class,
                        reproduction: serde_json::to_value(&attack).unwrap_or_default(),
                    },
                )?;
                // The finding is recorded as a violation too: it is a real
                // hole in a real guard, and the Warden's suite must grow to
                // cover it.
                let _ = landing_violation(&attack, evidence);
                self.deviant_win_streak += 1;
                self.warden_hold_streak = 0;
                first_time
            }
            Verdict::Repelled { .. } => {
                ledger.append(
                    Actor::Warden,
                    &Event::ExploitRepelled {
                        round: self.round,
                        class,
                    },
                )?;
                self.warden_hold_streak += 1;
                self.deviant_win_streak = 0;
                false
            }
            // Inert rounds move neither streak: nothing was learned about the
            // defence, so treating them as a Warden win would let the Deviant
            // manufacture a false sense of security by attacking badly.
            Verdict::Inert { .. } => false,
        };

        let containment = self.containment();
        let withstood: Vec<ExploitClass> = ExploitClass::ALL
            .iter()
            .copied()
            .filter(|c| !self.breached.contains(c))
            .collect();
        ledger.append(
            Actor::System,
            &Event::ContainmentMeasured {
                round: self.round,
                index: containment,
                withstood,
                breached: self.breached.iter().copied().collect(),
            },
        )?;

        Ok(Round {
            round: self.round,
            verdict,
            class,
            reward,
            novelty,
            containment,
            newly_breached,
        })
    }

    /// Whether the adversary has gone stale and needs a harder objective.
    ///
    /// A long Warden hold is *not* reassurance. In a coevolving system it
    /// usually means the Deviant stopped finding new angles, and a containment
    /// index that only ever climbs is the signature of an adversary that quit
    /// rather than a defence that won.
    pub fn adversary_is_stale(&self) -> bool {
        self.warden_hold_streak >= self.cfg.warden_hold_threshold
    }

    /// Whether the Warden may have regressed.
    ///
    /// A Deviant winning round after round usually means the defence got
    /// worse, not that the attacks got cleverer — the two look identical on
    /// the containment index and are told apart by the absolute yardstick.
    pub fn warden_may_have_regressed(&self) -> bool {
        self.deviant_win_streak >= self.cfg.deviant_runaway_threshold
    }
}
