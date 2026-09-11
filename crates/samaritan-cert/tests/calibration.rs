//! Tests for the certificate.
//!
//! The first one is the most important test in the repository. Every claim
//! this harness makes about self-improvement reduces to "the certificate does
//! not certify things that are not true, more than a fraction α of the time",
//! and that is not a claim to be reasoned about — it is a number, and it can
//! be measured.
//!
//! The simulation is seeded and deterministic, so a failure is reproducible
//! and a pass is not luck.

use samaritan_cert::{Betting, Certificate, Martingale, Pair, Provenance, Refusal, Spending};

/// xorshift64*, so the whole suite is reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
}

/// Paired outcomes where the candidate and incumbent are equally good.
///
/// `skill` is each arm's independent probability of solving a task, so
/// discordant pairs arise naturally and favour neither side. This is the null
/// the martingale is built against.
fn null_pairs(rng: &mut Rng, n: usize, skill: f64) -> Vec<Pair> {
    (0..n)
        .map(|_| Pair::new(rng.chance(skill), rng.chance(skill)))
        .collect()
}

/// Paired outcomes where the candidate is genuinely better.
fn improved_pairs(rng: &mut Rng, n: usize, incumbent: f64, candidate: f64) -> Vec<Pair> {
    (0..n)
        .map(|_| Pair::new(rng.chance(candidate), rng.chance(incumbent)))
        .collect()
}

// ================================================================== the claim

#[test]
fn a_null_candidate_is_certified_at_most_alpha_of_the_time() {
    // THE test. If this number is wrong, every other result in the project is
    // decoration: the machine would be committing changes that do nothing and
    // reporting them as improvements, and nothing downstream could tell.
    //
    // 2000 runs against a candidate that is exactly as good as the incumbent.
    // Ville bounds the probability of the wealth EVER crossing 1/α, so the
    // check is on the peak across the whole run, not on where it ended.
    const TRIALS: usize = 2000;
    const ALPHA: f64 = 0.05;
    const PAIRS: usize = 400;

    let mut rng = Rng::new(0x5A_4A_17_A4);
    let mut false_certifications = 0;

    for _ in 0..TRIALS {
        let pairs = null_pairs(&mut rng, PAIRS, 0.5);
        let mut m = Martingale::new(Betting::Adaptive { cap: 0.5 });
        m.observe_all(&pairs);
        if m.crossed(ALPHA) {
            false_certifications += 1;
        }
    }

    let rate = false_certifications as f64 / TRIALS as f64;
    // The bound is α. Allowing a little slack for simulation noise at this
    // sample size, but nothing like enough to hide a broken martingale: a
    // construction that is not a supermartingale runs far above this.
    assert!(
        rate <= ALPHA * 1.5,
        "false certification rate {rate:.4} exceeds the {ALPHA} guarantee \
         ({false_certifications}/{TRIALS})"
    );
    eprintln!("false certification rate: {rate:.4} against a bound of {ALPHA}");
}

#[test]
fn the_bound_holds_at_a_tighter_alpha_too() {
    const TRIALS: usize = 2000;
    const ALPHA: f64 = 0.01;
    let mut rng = Rng::new(99);
    let mut bad = 0;
    for _ in 0..TRIALS {
        let pairs = null_pairs(&mut rng, 400, 0.5);
        let mut m = Martingale::new(Betting::Adaptive { cap: 0.5 });
        m.observe_all(&pairs);
        if m.crossed(ALPHA) {
            bad += 1;
        }
    }
    let rate = bad as f64 / TRIALS as f64;
    assert!(rate <= ALPHA * 2.5, "rate {rate:.4} at alpha {ALPHA}");
}

#[test]
fn the_bound_holds_when_both_arms_are_weak() {
    // Low skill means mostly-failed tasks and few discordant pairs. The
    // guarantee must not depend on the tasks being easy.
    let mut rng = Rng::new(7);
    let mut bad = 0;
    for _ in 0..1000 {
        let pairs = null_pairs(&mut rng, 600, 0.15);
        let mut m = Martingale::new(Betting::Adaptive { cap: 0.5 });
        m.observe_all(&pairs);
        if m.crossed(0.05) {
            bad += 1;
        }
    }
    assert!(bad as f64 / 1000.0 <= 0.075, "{bad}/1000 at low skill");
}

#[test]
fn a_fixed_bet_is_also_a_valid_supermartingale() {
    // The adaptive rule is the default, but validity must not depend on it.
    let mut rng = Rng::new(4242);
    let mut bad = 0;
    for _ in 0..1500 {
        let pairs = null_pairs(&mut rng, 400, 0.5);
        let mut m = Martingale::new(Betting::Fixed(0.5));
        m.observe_all(&pairs);
        if m.crossed(0.05) {
            bad += 1;
        }
    }
    assert!(bad as f64 / 1500.0 <= 0.075, "{bad}/1500 with a fixed bet");
}

// ============================================================ it finds real gains

#[test]
fn a_genuinely_better_candidate_is_usually_certified() {
    // Power, not validity. A certificate that never fires is trivially safe
    // and completely useless, so the other half of the claim needs measuring
    // too.
    let mut rng = Rng::new(31337);
    let mut certified = 0;
    for _ in 0..200 {
        let pairs = improved_pairs(&mut rng, 400, 0.40, 0.60);
        let mut m = Martingale::new(Betting::Adaptive { cap: 0.5 });
        m.observe_all(&pairs);
        if m.crossed(0.05) {
            certified += 1;
        }
    }
    assert!(
        certified as f64 / 200.0 > 0.9,
        "only {certified}/200 real improvements were detected"
    );
}

#[test]
fn a_worse_candidate_is_essentially_never_certified() {
    let mut rng = Rng::new(5150);
    let mut certified = 0;
    for _ in 0..500 {
        // Incumbent 0.60, candidate 0.40: a clear regression.
        let pairs = improved_pairs(&mut rng, 400, 0.60, 0.40);
        let mut m = Martingale::new(Betting::Adaptive { cap: 0.5 });
        m.observe_all(&pairs);
        if m.crossed(0.05) {
            certified += 1;
        }
    }
    assert!(certified <= 2, "{certified}/500 regressions were certified");
}

// ==================================================== what it refuses to test

#[test]
fn a_sample_from_training_tasks_is_refused_outright() {
    // The failure that would invalidate everything while producing a
    // perfectly real-looking e-value.
    let mut rng = Rng::new(1);
    let pairs = improved_pairs(&mut rng, 400, 0.2, 0.9);
    let mut c = Certificate::new(Spending::default()).unwrap();
    assert_eq!(c.test(&pairs, Provenance::Train), Err(Refusal::Circular));
    assert_eq!(
        c.budget().tests(),
        0,
        "an invalid test must not consume budget"
    );
}

#[test]
fn too_few_informative_pairs_is_distinguished_from_weak_evidence() {
    // A candidate solving exactly what the incumbent solves yields no
    // discordant pairs at all, however many tasks are run. Reporting that as
    // "insufficient evidence" would send the search looking for more data
    // that does not exist.
    let pairs = vec![Pair::new(true, true); 500];
    let mut c = Certificate::new(Spending::default()).unwrap();
    match c.test(&pairs, Provenance::HeldOut) {
        Err(Refusal::TooFewDiscordant { discordant, .. }) => assert_eq!(discordant, 0),
        other => panic!("expected TooFewDiscordant, got {other:?}"),
    }
}

// ================================================================ the budget

#[test]
fn a_failed_test_still_costs_budget() {
    // Otherwise a search could propose indefinitely, testing until something
    // crossed by chance, which is exactly the multiple-comparisons problem
    // the budget exists to prevent.
    let mut rng = Rng::new(11);
    let mut c = Certificate::new(Spending::default()).unwrap();
    let before = c.budget().wealth();
    let pairs = null_pairs(&mut rng, 400, 0.5);
    let _ = c.test(&pairs, Provenance::HeldOut);
    assert!(c.budget().wealth() < before, "a failed test was free");
    assert_eq!(c.budget().tests(), 1);
}

#[test]
fn a_successful_commit_pays_back_into_the_budget() {
    // The property that makes investing usable where geometric spending is
    // not: a run that keeps finding real improvements keeps funding its own
    // testing.
    let mut rng = Rng::new(22);
    let mut c = Certificate::new(Spending::default()).unwrap();
    let start = c.budget().wealth();
    for _ in 0..5 {
        let pairs = improved_pairs(&mut rng, 500, 0.3, 0.7);
        c.test(&pairs, Provenance::HeldOut).expect("a real gain");
    }
    assert_eq!(c.budget().commits(), 5);
    assert!(
        c.budget().wealth() >= start * 0.5,
        "wealth collapsed to {} despite five real discoveries",
        c.budget().wealth()
    );
}

#[test]
fn a_run_of_failures_exhausts_the_budget_and_stops() {
    let mut rng = Rng::new(33);
    let mut c = Certificate::new(Spending::default()).unwrap();
    let mut refusals = 0;
    for _ in 0..200 {
        let pairs = null_pairs(&mut rng, 400, 0.5);
        if let Err(Refusal::BudgetExhausted) = c.test(&pairs, Provenance::HeldOut) {
            refusals += 1;
            break;
        }
    }
    assert_eq!(refusals, 1, "the budget never ran out");
    assert!(c.budget().exhausted());
}

#[test]
fn geometric_spending_stays_under_delta_in_total() {
    // The property that makes it right for level-3 rewrites, and the same
    // property that makes it useless at scale.
    let mut rng = Rng::new(44);
    let mut c = Certificate::new(Spending::Geometric { delta: 0.05 }).unwrap();
    for _ in 0..25 {
        let pairs = improved_pairs(&mut rng, 500, 0.3, 0.7);
        let _ = c.test(&pairs, Provenance::HeldOut);
    }
    assert!(
        c.budget().spent_total() <= 0.05 + 1e-9,
        "spent {} against a delta of 0.05",
        c.budget().spent_total()
    );
}

#[test]
fn geometric_spending_starves_which_is_why_it_is_not_the_default() {
    // Stated as a test so the tradeoff is on the record rather than buried in
    // a comment: after a few dozen tests the schedule can no longer clear any
    // realistic amount of evidence, however good the candidate is.
    let mut rng = Rng::new(55);
    let mut c = Certificate::new(Spending::Geometric { delta: 0.05 }).unwrap();
    let first = c.budget().next_alpha().unwrap();

    for _ in 0..25 {
        let pairs = improved_pairs(&mut rng, 400, 0.45, 0.55);
        let _ = c.test(&pairs, Provenance::HeldOut);
    }

    let later = c.budget().next_alpha().unwrap_or(0.0);
    assert!(
        later < first / 1e6,
        "expected the schedule to collapse: {first} -> {later}"
    );
}
