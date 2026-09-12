//! The runner: one full self-improvement run, configured as one experimental
//! arm, composed from every other crate.
//!
//! This is where the pieces meet. A round is:
//!
//! 1. **Level 0.** Play a batch of episodes on the training split with the
//!    current policy. Their outcomes go to the ledger and their tokens to the
//!    compute budget.
//! 2. **Reflect.** Mine the ledger into candidate mutations — evidence-backed,
//!    never model-authored ([`samaritan_reflect`]).
//! 3. **Search.** Run a level-1 NRPA search whose vocabulary is exactly the
//!    mined set, scored by playing more training episodes.
//! 4. **Certify.** Pair the search's candidate against the incumbent on the
//!    *held-out* split and test it with the anytime-valid certificate. Commit
//!    only if it crosses; either way, record it.
//! 5. **Measure.** Write the absolute yardstick (held-out utility), the
//!    compute spent, and — for the adversarial arm — one arena round and the
//!    containment index.
//!
//! The arm decides what the round contains beyond the shared spine:
//!
//! - `Solo` — the spine and nothing else. The control.
//! - `Critic` — a critic contributes extra candidates before the search:
//!   dense feedback without an adversary. Tests the design's central
//!   question, whether the useful ingredient in adversarial training is the
//!   antagonism or merely the signal.
//! - `Adversarial` — the Deviant runs a round against the guards and the
//!   containment index is measured.
//!
//! Nothing here trusts a single measurement. Containment is relative and can
//! rise while capability rots; the yardstick is absolute and catches that.
//! Both are written every round, and [`samaritan_ledger::diagnose`] reads the
//! pair.

pub mod patch;
pub mod playfield;
pub mod propose;

use samaritan_cert::{Certificate, Pair, Provenance, Spending};
use samaritan_corpus::Split;
use samaritan_dsl::{Digest, Mutation};
use samaritan_kernel::{Admission, ComputeBudget, ComputeTier};
use samaritan_ledger::{Actor, Arm, Event, Ledger, LedgerError};
use samaritan_reflect::{mine, Candidate, Corpus, MineConfig};
use samaritan_search::{Budget as SearchBudget, PolicyState};
use serde::{Deserialize, Serialize};

pub use playfield::{summarise, BatchSummary, Playfield};

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("ledger: {0}")]
    Ledger(#[from] LedgerError),
    #[error("certificate budget could not be created: {0}")]
    Cert(#[from] samaritan_cert::BudgetError),
}

/// How a run is configured. Declared once, up front, and written to the
/// ledger before anything is measured — the comparative question is
/// unanswerable if the arm is decided after seeing the results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunConfig {
    pub arm: Arm,
    /// The x-axis for "faster". Rounds are not comparable across arms; an
    /// adversarial arm runs two agents and would otherwise take a free
    /// compute advantage.
    pub compute_budget_tokens: u64,
    pub seed: u64,
    /// Pins the corpus the run trained on, so a transfer claim can be checked
    /// against what was actually seen.
    pub corpus_manifest: Digest,
    /// Episodes per level-0 batch.
    pub batch_size: usize,
    /// Held-out tasks the certificate pairs over.
    pub certify_pairs: usize,
    /// Held-out tasks the yardstick averages over.
    pub yardstick_tasks: usize,
    pub search_budget: SearchBudget,
    pub mine: MineConfig,
    pub certify: Spending,
}

impl RunConfig {
    pub fn new(arm: Arm, compute_budget_tokens: u64, seed: u64, corpus_manifest: Digest) -> Self {
        Self {
            arm,
            compute_budget_tokens,
            seed,
            corpus_manifest,
            batch_size: 32,
            certify_pairs: 60,
            yardstick_tasks: 50,
            search_budget: SearchBudget {
                iterations: vec![0, 40],
                alpha: 1.0,
            },
            mine: MineConfig::default(),
            certify: Spending::default(),
        }
    }
}

/// A source of extra candidates for the `Critic` arm.
///
/// A critic finds flaws and reports them as lessons; it does not exploit them.
/// Modelling it as an extra candidate source — rather than an adversary — is
/// exactly the antagonism/signal-density separation the experiment exists to
/// test. Its candidates are mined-shaped too: evidence-backed, not authored.
pub trait Critic {
    fn review(&mut self, corpus: &Corpus) -> Vec<Candidate>;
}

/// What one round came to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundReport {
    pub round: u64,
    pub yardstick: f64,
    pub committed: Option<Mutation>,
    pub candidates_considered: usize,
    pub tokens_spent: u64,
    pub compute_tier: ComputeTier,
    /// Present only for the adversarial arm.
    pub containment: Option<f64>,
}

/// One self-improvement run.
pub struct Runner {
    cfg: RunConfig,
    policy: PolicyState,
    admission: Admission,
    budget: ComputeBudget,
    certificate: Certificate,
    round: u64,
    /// The adversary's arena, for the adversarial arm only.
    arena: Option<samaritan_adversary::Arena>,
}

impl Runner {
    pub fn new(cfg: RunConfig) -> Result<Self, RunError> {
        let arena = matches!(cfg.arm, Arm::Adversarial)
            .then(|| samaritan_adversary::Arena::new(Default::default()));
        Ok(Self {
            budget: ComputeBudget::new(cfg.compute_budget_tokens),
            certificate: Certificate::new(cfg.certify)?,
            admission: Admission::default(),
            policy: PolicyState::default(),
            round: 0,
            arena,
            cfg,
        })
    }

    pub fn policy(&self) -> &PolicyState {
        &self.policy
    }

    pub fn budget(&self) -> &ComputeBudget {
        &self.budget
    }

    pub fn round(&self) -> u64 {
        self.round
    }

    /// Declare the run. Must be called once, before any round.
    pub fn configure(&mut self, ledger: &mut Ledger) -> Result<(), RunError> {
        ledger.append(
            Actor::System,
            &Event::RunConfigured {
                arm: self.cfg.arm,
                compute_budget_tokens: self.cfg.compute_budget_tokens,
                seed: self.cfg.seed,
                corpus_manifest: self.cfg.corpus_manifest,
            },
        )?;
        Ok(())
    }

    /// Whether another round can be afforded.
    pub fn can_continue(&self) -> bool {
        self.budget.tier().allows_work()
    }

    /// Run one round. Returns `None` once the compute budget is spent, so a
    /// caller can `while let Some(r) = runner.step(...)`.
    pub fn step(
        &mut self,
        field: &mut dyn Playfield,
        critic: Option<&mut dyn Critic>,
        deviant: Option<&mut dyn samaritan_adversary::Attacker>,
        ledger: &mut Ledger,
    ) -> Result<Option<RoundReport>, RunError> {
        if !self.budget.tier().allows_work() {
            return Ok(None);
        }
        self.round += 1;
        ledger.append(Actor::System, &Event::RoundOpened { round: self.round })?;

        // ---- level 0: play a training batch with the current policy --------
        let batch = field.play(&self.policy, Split::Train, self.cfg.batch_size);
        let summary = summarise(&batch);
        self.spend(summary.tokens, batch.len() as u32, ledger)?;
        self.record_outcomes(&batch, ledger)?;

        // ---- reflect: mine candidates from this batch + the ledger ---------
        // Calibration comes from the episodes just run; breaches come from the
        // durable record. Mining from the batch in hand keeps the reflection
        // pass independent of whether the episode layer logged per-decision.
        let corpus = Corpus::from_predictions(&summary.calibration, ledger)?;
        let mut candidates = mine(&corpus, &self.cfg.mine);

        // The critic arm adds dense feedback here, before the search. Its
        // candidates compete with the mined ones on the same footing.
        if let Some(critic) = critic {
            candidates.extend(critic.review(&corpus));
        }
        let candidates_considered = candidates.len();

        // ---- search + certify: only self-modify when affordable ------------
        let mut committed = None;
        if self.budget.tier().allows_self_modification() && !candidates.is_empty() {
            committed = self.search_and_certify(candidates, field, ledger)?;
        }

        // ---- the arms race, for the adversarial arm ------------------------
        let containment = self.maybe_arena_round(deviant, ledger)?;

        // ---- measure the absolute yardstick every round --------------------
        let held = field.play(&self.policy, Split::HeldOut, self.cfg.yardstick_tasks);
        let yardstick = summarise(&held).mean_utility;
        self.spend(summarise(&held).tokens, held.len() as u32, ledger)?;
        ledger.append(
            Actor::System,
            &Event::YardstickMeasured {
                round: self.round,
                utility: yardstick,
                tasks: held.len() as u32,
            },
        )?;

        Ok(Some(RoundReport {
            round: self.round,
            yardstick,
            committed,
            candidates_considered,
            tokens_spent: self.budget.spent(),
            compute_tier: self.budget.tier(),
            containment,
        }))
    }

    /// The search over mined candidates, then the certificate gate.
    fn search_and_certify(
        &mut self,
        candidates: Vec<Candidate>,
        field: &mut dyn Playfield,
        ledger: &mut Ledger,
    ) -> Result<Option<Mutation>, RunError> {
        // The search scores a policy by playing training episodes with it.
        // Held-out tasks are never touched here — they belong to the
        // certificate, and spending them on the search would be the exact
        // circularity the certificate refuses.
        let base_for_search = self.policy.clone();
        let admission = self.admission.clone();
        let batch = self.cfg.batch_size;
        let (_score, mutations) = samaritan_reflect::search_mined(
            candidates,
            admission,
            self.cfg.seed.wrapping_add(self.round),
            4,
            &self.cfg.search_budget,
            |candidate_state: &PolicyState| {
                // Overlay the searched mutations onto the *run's* current
                // policy, then score.
                let mut merged = base_for_search.clone();
                merge_lessons(&mut merged, candidate_state, &self.admission);
                let out = field.play(&merged, Split::Train, batch.min(8));
                summarise(&out).mean_utility
            },
        );

        if mutations.is_empty() {
            return Ok(None);
        }

        // Build the candidate policy: the incumbent plus the searched changes.
        let mut candidate_policy = self.policy.clone();
        for m in &mutations {
            let _ = candidate_policy.apply(m, &self.admission);
        }

        // Pair candidate against incumbent on held-out tasks.
        let paired = field.play_paired(
            &self.policy,
            &candidate_policy,
            Split::HeldOut,
            self.cfg.certify_pairs,
        );
        let pairs: Vec<Pair> = paired
            .iter()
            .map(|(c, i)| Pair::new(*c, *i))
            .collect();

        let first = mutations[0].clone();
        match self.certificate.test(&pairs, Provenance::HeldOut) {
            Ok(granted) => {
                // Commit: the candidate policy becomes the incumbent.
                self.policy = candidate_policy;
                ledger.append(
                    Actor::Warden,
                    &Event::MutationCommitted {
                        mutation: first.clone(),
                        evidence: granted.e_value,
                        alpha_spent: granted.alpha,
                        to_policy: self.policy.digest(),
                    },
                )?;
                Ok(Some(first))
            }
            Err(refusal) => {
                // A failed candidate still buys information, so it is recorded
                // rather than dropped.
                ledger.append(
                    Actor::Warden,
                    &Event::MutationAbandoned {
                        mutation: first,
                        evidence: 0.0,
                        reason: refusal.to_string(),
                    },
                )?;
                Ok(None)
            }
        }
    }

    /// One arena round, for the adversarial arm. `None` for every other arm.
    fn maybe_arena_round(
        &mut self,
        deviant: Option<&mut dyn samaritan_adversary::Attacker>,
        ledger: &mut Ledger,
    ) -> Result<Option<f64>, RunError> {
        let Some(arena) = self.arena.as_mut() else {
            return Ok(None);
        };
        let known = arena.breached();
        let policy = samaritan_dsl::MutationPolicy::new();

        // The live Deviant proposes an attack; with none supplied, the seeded
        // opening book stands in, re-throwing known attacks at the real
        // guards. Either way the attack is *data* evaluated against the
        // guards — the model never runs on the host.
        let attack = match deviant {
            Some(d) => d.propose(&known, &policy),
            None => {
                let book = samaritan_adversary::opening_book();
                book[(self.round as usize) % book.len()].clone()
            }
        };
        let verdict = samaritan_adversary::attempt(&attack, &arena_target());
        let round = arena.record(&attack, verdict, ledger)?;
        Ok(Some(round.containment))
    }

    fn spend(&mut self, tokens: u64, episodes: u32, ledger: &mut Ledger) -> Result<(), RunError> {
        self.budget.spend(tokens);
        ledger.append(
            Actor::System,
            &Event::ComputeSpent {
                round: self.round,
                tokens,
                calls: episodes,
            },
        )?;
        Ok(())
    }

    fn record_outcomes(
        &self,
        batch: &[samaritan_episode::EpisodeOutcome],
        ledger: &mut Ledger,
    ) -> Result<(), RunError> {
        // The episode crate writes the per-decision detail; here the runner
        // only needs the containment/experiment-level rows, which the episode
        // does not know about. Nothing to add per outcome at this level today,
        // but the hook exists so a future signal has an obvious home.
        let _ = (batch, ledger);
        Ok(())
    }
}

/// Copy any lessons present in `from` but not in `into`, through the applier
/// so the frozen rules still gate them.
fn merge_lessons(into: &mut PolicyState, from: &PolicyState, admission: &Admission) {
    for l in from.lessons() {
        if !into.lessons().iter().any(|x| x.text == l.text) {
            let _ = into.apply(
                &Mutation::LessonAdd {
                    text: l.text.clone(),
                },
                admission,
            );
        }
    }
}

/// A fresh, intact set of guards for the arena to attack.
///
/// Built anew each round rather than held, so an attack cannot leave residue
/// that flatters the next round's defence.
fn arena_target() -> samaritan_adversary::Target {
    let sandbox = std::env::temp_dir().join("samaritan-arena-target");
    let _ = std::fs::create_dir_all(&sandbox);
    samaritan_adversary::Target {
        admission: Admission::default(),
        ceiling: samaritan_kernel::AutonomyCeiling::closed(),
        promotions: samaritan_kernel::Promotions::none(),
        budget: ComputeBudget::new(1_000_000),
        sandbox,
    }
}
