//! Integration tests for the runner.
//!
//! The whole point of the `Playfield` seam is on display here: a run is driven
//! end to end — budget, reflection, search, the certificate gate, the arms
//! race, the yardstick — with a scripted playfield and no model. If this
//! passes, the orchestration is correct; whether the *model* can actually
//! solve anything is a separate question the harness cannot answer on a
//! machine it must not heat up.

use samaritan_corpus::{Split, TaskId};
use samaritan_dsl::Digest;
use samaritan_episode::{Ending, EpisodeOutcome};
use samaritan_kernel::{Components, EpisodeUtility, Weights};
use samaritan_ledger::{Arm, FixedClock, Ledger};
use samaritan_run::{Playfield, RunConfig, Runner};
use samaritan_search::PolicyState;

fn ledger() -> Ledger {
    Ledger::in_memory(Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap()
}

/// Build an outcome with a given success, confidence, and token cost.
fn outcome(solved: bool, confidence: f64, tokens: u64) -> EpisodeOutcome {
    let components = Components {
        task_success: if solved { 1.0 } else { 0.0 },
        brier: 0.0,
        approvals_requested: 0,
        seconds: 1.0,
    };
    EpisodeOutcome {
        task: TaskId("t".into()),
        ending: if solved {
            Ending::Solved { at_step: 1 }
        } else {
            Ending::StepsExhausted
        },
        utility: EpisodeUtility::clean(&components, &Weights::default()),
        components,
        steps: 1,
        predictions: vec![(confidence, solved)],
        tokens,
        violations: vec![],
        answer: None,
        trace: None,
    }
}

/// A playfield whose success rate depends on how many lessons the policy
/// carries — so "learning" (applying mined lessons) genuinely raises the
/// score, and the certificate has something real to certify.
struct LearningField {
    /// Base solve probability with no lessons, and the lift per lesson.
    base: f64,
    lift_per_lesson: f64,
    tokens_per_episode: u64,
    rng: u64,
}

impl LearningField {
    fn new(base: f64, lift: f64) -> Self {
        Self {
            base,
            lift_per_lesson: lift,
            tokens_per_episode: 1000,
            rng: 1,
        }
    }
    fn unit(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn solve_p(&self, policy: &PolicyState) -> f64 {
        (self.base + self.lift_per_lesson * policy.lessons().len() as f64).clamp(0.0, 0.99)
    }
}

impl Playfield for LearningField {
    fn play(&mut self, policy: &PolicyState, _split: Split, n: usize) -> Vec<EpisodeOutcome> {
        let p = self.solve_p(policy);
        (0..n)
            .map(|_| {
                let solved = self.unit() < p;
                // Overconfident on purpose when there are no lessons, so the
                // miner has a calibration pattern to find: high confidence,
                // frequent misses.
                let conf = if policy.lessons().is_empty() { 0.9 } else { 0.6 };
                outcome(solved, conf, self.tokens_per_episode)
            })
            .collect()
    }

    fn play_paired(
        &mut self,
        incumbent: &PolicyState,
        candidate: &PolicyState,
        _split: Split,
        n: usize,
    ) -> Vec<(bool, bool)> {
        let pi = self.solve_p(incumbent);
        let pc = self.solve_p(candidate);
        (0..n)
            .map(|_| (self.unit() < pc, self.unit() < pi))
            .collect()
    }
}

fn config(arm: Arm) -> RunConfig {
    let mut c = RunConfig::new(arm, 200_000, 7, Digest::ZERO);
    c.batch_size = 40;
    c.certify_pairs = 120;
    c.yardstick_tasks = 30;
    c
}

// ===================================================== the spine

#[test]
fn a_run_is_declared_before_it_is_measured() {
    let mut l = ledger();
    let mut r = Runner::new(config(Arm::Solo)).unwrap();
    r.configure(&mut l).unwrap();
    let rows = l.by_kind("run_configured").unwrap();
    assert_eq!(rows.len(), 1, "the arm must be on the record before round one");
}

#[test]
fn a_round_measures_the_yardstick_and_spends_compute() {
    let mut l = ledger();
    let mut r = Runner::new(config(Arm::Solo)).unwrap();
    r.configure(&mut l).unwrap();
    let mut field = LearningField::new(0.4, 0.1);

    let report = r.step(&mut field, None, None, &mut l).unwrap().expect("a round ran");
    assert_eq!(report.round, 1);
    assert!(report.tokens_spent > 0, "episodes cost tokens");
    assert_eq!(l.by_kind("yardstick_measured").unwrap().len(), 1);
    assert!(l.by_kind("compute_spent").unwrap().len() >= 1);
    assert!(l.by_kind("round_opened").unwrap().len() == 1);
    l.verify().unwrap();
}

#[test]
fn the_budget_stops_the_run() {
    // A tiny budget must halt the loop, and `step` must report exhaustion by
    // returning None rather than running a round it cannot pay for.
    let mut l = ledger();
    let mut cfg = config(Arm::Solo);
    cfg.compute_budget_tokens = 5_000; // a couple of batches at most
    let mut r = Runner::new(cfg).unwrap();
    r.configure(&mut l).unwrap();
    let mut field = LearningField::new(0.4, 0.1);

    let mut rounds = 0;
    while let Some(_report) = r.step(&mut field, None, None, &mut l).unwrap() {
        rounds += 1;
        assert!(rounds < 100, "the budget never stopped the run");
    }
    assert!(rounds >= 1, "at least one round should fit");
    assert!(!r.can_continue(), "the run ended because compute ran out");
    l.verify().unwrap();
}

// ============================================ the certificate gate

#[test]
fn a_real_improvement_is_eventually_committed() {
    // Lessons genuinely lift the solve rate here, so a mined lesson paired on
    // held-out tasks should clear the certificate and be committed. This is
    // the whole loop working: reflect -> search -> certify -> commit.
    let mut l = ledger();
    let mut r = Runner::new(config(Arm::Solo)).unwrap();
    r.configure(&mut l).unwrap();
    // Strong lift, so the signal is unambiguous and the test is not flaky.
    let mut field = LearningField::new(0.30, 0.30);

    let mut committed = 0;
    for _ in 0..8 {
        if r.step(&mut field, None, None, &mut l).unwrap().is_none() {
            break;
        }
        committed = l.by_kind("mutation_committed").unwrap().len();
        if committed > 0 {
            break;
        }
    }
    assert!(committed > 0, "a genuinely better lesson was never committed");
    assert!(
        !r.policy().lessons().is_empty(),
        "the committed lesson should be in the live policy"
    );
    l.verify().unwrap();
}

#[test]
fn a_policy_that_learns_nothing_commits_nothing() {
    // When lessons do not help (zero lift), the certificate must refuse to
    // commit them however many rounds run. Committing here would be the false
    // certification the whole cert crate exists to bound.
    let mut l = ledger();
    let mut r = Runner::new(config(Arm::Solo)).unwrap();
    r.configure(&mut l).unwrap();
    let mut field = LearningField::new(0.5, 0.0); // lessons do nothing

    for _ in 0..6 {
        if r.step(&mut field, None, None, &mut l).unwrap().is_none() {
            break;
        }
    }
    assert_eq!(
        l.by_kind("mutation_committed").unwrap().len(),
        0,
        "a useless lesson was committed"
    );
}

#[test]
fn self_modification_stops_when_compute_is_critical() {
    // Below the critical tier, task work continues but self-modification is
    // switched off. Spend most of the budget, then confirm a round still runs
    // (yardstick measured) but commits nothing.
    let mut l = ledger();
    let mut cfg = config(Arm::Solo);
    cfg.compute_budget_tokens = 100_000;
    let mut r = Runner::new(cfg).unwrap();
    r.configure(&mut l).unwrap();
    let mut field = LearningField::new(0.3, 0.3);

    // Burn down to the critical band.
    while r.budget().tier().allows_self_modification() {
        if r.step(&mut field, None, None, &mut l).unwrap().is_none() {
            break;
        }
    }
    let commits_before = l.by_kind("mutation_committed").unwrap().len();

    // A round in the critical band still measures the yardstick.
    if r.can_continue() {
        let before_yard = l.by_kind("yardstick_measured").unwrap().len();
        r.step(&mut field, None, None, &mut l).unwrap();
        assert!(
            l.by_kind("yardstick_measured").unwrap().len() > before_yard,
            "task work should continue in the critical band"
        );
        assert_eq!(
            l.by_kind("mutation_committed").unwrap().len(),
            commits_before,
            "self-modification must stop before task work does"
        );
    }
}

// ==================================================== the arms

#[test]
fn the_adversarial_arm_measures_containment_every_round() {
    let mut l = ledger();
    let mut r = Runner::new(config(Arm::Adversarial)).unwrap();
    r.configure(&mut l).unwrap();
    let mut field = LearningField::new(0.4, 0.1);

    let report = r.step(&mut field, None, None, &mut l).unwrap().unwrap();
    assert!(report.containment.is_some(), "the adversarial arm runs the arena");
    assert_eq!(report.containment, Some(1.0), "intact guards repel the opening book");
    assert_eq!(l.by_kind("containment_measured").unwrap().len(), 1);
}

#[test]
fn the_solo_arm_runs_no_arena() {
    let mut l = ledger();
    let mut r = Runner::new(config(Arm::Solo)).unwrap();
    r.configure(&mut l).unwrap();
    let mut field = LearningField::new(0.4, 0.1);

    let report = r.step(&mut field, None, None, &mut l).unwrap().unwrap();
    assert_eq!(report.containment, None, "the control arm has no adversary");
    assert_eq!(l.by_kind("containment_measured").unwrap().len(), 0);
}

/// A critic that always proposes one extra lesson, standing in for dense
/// feedback without antagonism.
struct AlwaysCritic;
impl samaritan_run::Critic for AlwaysCritic {
    fn review(&mut self, _corpus: &samaritan_reflect::Corpus) -> Vec<samaritan_reflect::Candidate> {
        vec![samaritan_reflect::Candidate {
            mutation: samaritan_dsl::Mutation::LessonAdd {
                text: "critic: name the expected value from the assertion before editing".into(),
            },
            rationale: "critic feedback".into(),
            support: 100,
            effect: 0.9,
        }]
    }
}

#[test]
fn the_critic_arm_contributes_extra_candidates() {
    // The critic's lesson should widen the candidate set the search considers,
    // even on a round where mining alone would have found nothing.
    let mut l = ledger();
    let mut r = Runner::new(config(Arm::Critic)).unwrap();
    r.configure(&mut l).unwrap();
    let mut field = LearningField::new(0.5, 0.3); // calibrated, so mining is quiet
    let mut critic = AlwaysCritic;

    let report = r.step(&mut field, Some(&mut critic), None, &mut l).unwrap().unwrap();
    assert!(
        report.candidates_considered >= 1,
        "the critic should have contributed a candidate even when mining was silent"
    );
}

#[test]
fn the_whole_run_stays_verifiable() {
    // Every arm, several rounds, and the hash chain must still verify at the
    // end -- the ledger is the experiment's only record, and a run that
    // corrupts it has measured nothing.
    for arm in [Arm::Solo, Arm::Adversarial, Arm::Critic] {
        let mut l = ledger();
        let mut r = Runner::new(config(arm)).unwrap();
        r.configure(&mut l).unwrap();
        let mut field = LearningField::new(0.4, 0.2);
        let mut critic = AlwaysCritic;
        for _ in 0..4 {
            let c: Option<&mut dyn samaritan_run::Critic> =
                if arm == Arm::Critic { Some(&mut critic) } else { None };
            if r.step(&mut field, c, None, &mut l).unwrap().is_none() {
                break;
            }
        }
        l.verify().unwrap();
    }
}
