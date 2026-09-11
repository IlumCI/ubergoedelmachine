//! Tests for the experiment instruments.
//!
//! The claim these have to defend: containment alone cannot tell mutual
//! improvement from mutual degradation, and the pair of measures can. Most of
//! what follows constructs a run that looks healthy on the relative measure
//! and is not.

use samaritan_dsl::Digest;
use samaritan_ledger::{
    Actor, Arm, Event, ExploitClass, FixedClock, Ledger, Verdict, diagnose,
};

fn mem() -> Ledger {
    Ledger::in_memory(Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap()
}

/// A run where capability climbs.
fn rising() -> Vec<(u64, f64)> {
    vec![(1, 0.40), (2, 0.45), (3, 0.51), (4, 0.58)]
}

/// A run where capability rots.
fn falling() -> Vec<(u64, f64)> {
    vec![(1, 0.58), (2, 0.51), (3, 0.45), (4, 0.40)]
}

fn flat() -> Vec<(u64, f64)> {
    vec![(1, 0.50), (2, 0.501), (3, 0.499), (4, 0.50)]
}

const EPS: f64 = 0.005;

// ------------------------------------------------------- the four outcomes

#[test]
fn rising_capability_is_improvement_however_containment_moves() {
    assert_eq!(diagnose(&rising(), &flat(), EPS), Verdict::Improving);
    assert_eq!(diagnose(&rising(), &falling(), EPS), Verdict::Improving);
    assert_eq!(diagnose(&rising(), &rising(), EPS), Verdict::Improving);
}

#[test]
fn the_degenerate_attractor_is_named_not_missed() {
    // The case the whole instrument exists for: the Deviant and the Warden
    // are both optimising the game, containment looks excellent, and the
    // agent is getting worse at its actual job.
    assert_eq!(
        diagnose(&falling(), &rising(), EPS),
        Verdict::Degenerate,
        "capability falling while containment improves is the failure mode"
    );
    assert_eq!(
        diagnose(&falling(), &flat(), EPS),
        Verdict::Degenerate,
        "flat containment hides it just as well as rising containment"
    );
}

#[test]
fn containment_alone_would_have_called_that_a_success() {
    // Stated as an executable claim rather than a comment. On the relative
    // measure the degenerate run is indistinguishable from the healthy one.
    let healthy = diagnose(&rising(), &rising(), EPS);
    let degenerate = diagnose(&falling(), &rising(), EPS);

    let containment_only = |c: &[(u64, f64)]| c.last().unwrap().1 >= c.first().unwrap().1;
    assert!(containment_only(&rising()));
    assert!(
        containment_only(&rising()),
        "both runs have identical containment trends"
    );
    assert_ne!(
        healthy, degenerate,
        "but the yardstick separates them completely"
    );
}

#[test]
fn losing_on_both_axes_is_collapse_not_degeneracy() {
    assert_eq!(diagnose(&falling(), &falling(), EPS), Verdict::Collapsing);
}

#[test]
fn going_nowhere_is_reported_as_disengagement() {
    assert_eq!(diagnose(&flat(), &flat(), EPS), Verdict::Stagnant);
}

#[test]
fn one_round_yields_no_verdict_rather_than_a_guess() {
    assert_eq!(diagnose(&[], &[], EPS), Verdict::InsufficientData);
    assert_eq!(diagnose(&[(1, 0.5)], &[], EPS), Verdict::InsufficientData);
}

#[test]
fn the_solo_arm_needs_no_containment_series() {
    // The control arm has no adversary, so containment is absent. The
    // yardstick alone must still produce a verdict.
    assert_eq!(diagnose(&rising(), &[], EPS), Verdict::Improving);
    assert_eq!(diagnose(&falling(), &[], EPS), Verdict::Degenerate);
}

#[test]
fn noise_below_the_threshold_does_not_read_as_decline() {
    let jittery = vec![(1, 0.500), (2, 0.498), (3, 0.502), (4, 0.499)];
    assert_eq!(
        diagnose(&jittery, &flat(), EPS),
        Verdict::Stagnant,
        "a noisy flat run must not be reported as degenerate"
    );
}

// -------------------------------------------------------------- the ledger

#[test]
fn yardstick_and_transfer_are_recorded_separately_per_corpus() {
    let mut l = mem();
    for (round, y, t_rust, t_py) in [(1u64, 0.40, 0.30, 0.22), (2, 0.55, 0.31, 0.23)] {
        l.append(
            Actor::System,
            &Event::YardstickMeasured {
                round,
                utility: y,
                tasks: 200,
            },
        )
        .unwrap();
        for (corpus, u) in [("rust-unseen", t_rust), ("python-unseen", t_py)] {
            l.append(
                Actor::System,
                &Event::TransferMeasured {
                    round,
                    corpus: corpus.into(),
                    utility: u,
                    tasks: 100,
                },
            )
            .unwrap();
        }
    }

    assert_eq!(l.yardstick_history().unwrap(), vec![(1, 0.40), (2, 0.55)]);
    assert_eq!(
        l.transfer_history("rust-unseen").unwrap(),
        vec![(1, 0.30), (2, 0.31)]
    );
    assert_eq!(
        l.transfer_history("python-unseen").unwrap(),
        vec![(1, 0.22), (2, 0.23)]
    );
    assert!(l.transfer_history("never-measured").unwrap().is_empty());
}

#[test]
fn memorisation_shows_up_as_a_yardstick_transfer_gap() {
    // The distinction the transfer corpus buys: the agent got much better on
    // repositories it trained against and barely moved on ones it did not.
    let mut l = mem();
    for (round, y, t) in [(1u64, 0.40, 0.300), (2, 0.60, 0.305), (3, 0.80, 0.302)] {
        l.append(
            Actor::System,
            &Event::YardstickMeasured {
                round,
                utility: y,
                tasks: 200,
            },
        )
        .unwrap();
        l.append(
            Actor::System,
            &Event::TransferMeasured {
                round,
                corpus: "unseen".into(),
                utility: t,
                tasks: 100,
            },
        )
        .unwrap();
    }

    assert_eq!(
        diagnose(&l.yardstick_history().unwrap(), &[], EPS),
        Verdict::Improving
    );
    assert_eq!(
        diagnose(&l.transfer_history("unseen").unwrap(), &[], EPS),
        Verdict::Stagnant,
        "improvement that does not transfer is memorisation, and must be visible as such"
    );
}

#[test]
fn compute_is_accumulated_so_arms_can_be_matched_fairly() {
    // Rounds are not comparable across arms — the adversarial arm runs two
    // agents. Tokens are the honest x-axis for any claim about speed.
    let mut l = mem();
    l.append(
        Actor::System,
        &Event::RunConfigured {
            arm: Arm::Adversarial,
            compute_budget_tokens: 10_000_000,
            seed: 42,
            corpus_manifest: Digest::of_bytes(b"corpus-v1"),
        },
    )
    .unwrap();
    for round in 1..=3u64 {
        l.append(
            Actor::System,
            &Event::ComputeSpent {
                round,
                tokens: 1_000,
                calls: 10,
            },
        )
        .unwrap();
    }
    assert_eq!(l.compute_spent().unwrap(), (3_000, 30));
}

#[test]
fn the_arm_is_declared_before_anything_is_measured() {
    let mut l = mem();
    l.append(
        Actor::System,
        &Event::RunConfigured {
            arm: Arm::Solo,
            compute_budget_tokens: 1,
            seed: 7,
            corpus_manifest: Digest::ZERO,
        },
    )
    .unwrap();
    let e = &l.entries().unwrap()[0];
    assert_eq!(e.event.kind(), "run_configured");
    match e.event {
        Event::RunConfigured { arm, seed, .. } => {
            assert_eq!(arm, Arm::Solo);
            assert_eq!(seed, 7);
        }
        _ => panic!("wrong event"),
    }
}

#[test]
fn attack_transfer_and_reference_security_round_trip() {
    let mut l = mem();
    l.append(
        Actor::Deviant,
        &Event::AttackTransfer {
            round: 4,
            class: ExploitClass::TierMisgrade,
            target: "warden@round1".into(),
            succeeded: true,
        },
    )
    .unwrap();
    l.append(
        Actor::System,
        &Event::ReferenceAdversaryResult {
            round: 4,
            withstood: 9,
            attempted: 10,
        },
    )
    .unwrap();

    assert_eq!(l.by_kind("attack_transfer").unwrap().len(), 1);
    match l.by_kind("reference_adversary_result").unwrap()[0].event {
        Event::ReferenceAdversaryResult {
            withstood,
            attempted,
            ..
        } => assert_eq!((withstood, attempted), (9, 10)),
        _ => panic!("wrong event"),
    }
    assert_eq!(l.verify().unwrap(), 2);
}

#[test]
fn generalisation_of_a_patch_is_recorded_as_a_ratio_not_a_boolean() {
    // A patch that survives 1 of 20 perturbations "closed" the class only in
    // the narrowest sense, and the row has to be able to say so.
    let mut l = mem();
    l.append(
        Actor::Warden,
        &Event::PatchGeneralized {
            round: 2,
            class: ExploitClass::AdmissionBypass,
            perturbations_tested: 20,
            perturbations_survived: 1,
        },
    )
    .unwrap();
    match l.by_kind("patch_generalized").unwrap()[0].event {
        Event::PatchGeneralized {
            perturbations_tested,
            perturbations_survived,
            ..
        } => {
            assert_eq!(perturbations_tested, 20);
            assert_eq!(perturbations_survived, 1);
        }
        _ => panic!("wrong event"),
    }
}

// ------------------------------------------------------------- frontier

#[test]
fn frontier_progress_keeps_unsolved_tasks_as_censored_not_as_zeros() {
    // The statistical point. "Attempted 400 times, never solved" is a
    // right-censored observation — the true time-to-solve is *greater than*
    // the tokens spent. Scoring it zero, or dropping it, turns a survival
    // comparison into a meaningless average.
    let mut l = mem();
    for (task, tokens) in [("ctx-overflow", 5_000u64), ("many-step", 9_000)] {
        l.append(
            Actor::Warden,
            &Event::FrontierAttempted {
                round: 1,
                task: task.into(),
                solved: false,
                tokens_spent: tokens,
            },
        )
        .unwrap();
    }
    l.append(
        Actor::Warden,
        &Event::FrontierAttempted {
            round: 4,
            task: "ctx-overflow".into(),
            solved: true,
            tokens_spent: 3_000,
        },
    )
    .unwrap();
    l.append(
        Actor::Warden,
        &Event::FrontierSolved {
            round: 4,
            task: "ctx-overflow".into(),
            tokens_to_first_solve: 8_000,
            mutations_committed: 3,
        },
    )
    .unwrap();

    let p = l.frontier_progress().unwrap();
    assert_eq!(p.solved, vec![("ctx-overflow".to_string(), 8_000)]);
    assert_eq!(
        p.censored,
        vec![("many-step".to_string(), 9_000)],
        "the unsolved task must survive as a censored observation"
    );
    assert_eq!(p.solve_rate(), Some(0.5));
    assert!(p.any_solved());
}

#[test]
fn a_later_duplicate_solve_does_not_overwrite_the_first() {
    let mut l = mem();
    for tokens in [12_000u64, 8_000, 20_000] {
        l.append(
            Actor::Warden,
            &Event::FrontierSolved {
                round: 9,
                task: "t".into(),
                tokens_to_first_solve: tokens,
                mutations_committed: 1,
            },
        )
        .unwrap();
    }
    assert_eq!(
        l.frontier_progress().unwrap().solved,
        vec![("t".to_string(), 8_000)],
        "first solve means earliest, however the rows were written"
    );
}

#[test]
fn an_untouched_frontier_reports_nothing_rather_than_success() {
    let l = mem();
    let p = l.frontier_progress().unwrap();
    assert_eq!(p.solve_rate(), None, "no attempts is not a 0% or a 100%");
    assert!(!p.any_solved());
}

#[test]
fn a_solve_with_no_self_modification_is_visible_as_such() {
    // If a frontier task falls with zero committed mutations, it was reachable
    // by the base scaffold all along and was never a frontier task. The row
    // has to make that checkable rather than letting it read as a triumph.
    let mut l = mem();
    l.append(
        Actor::Warden,
        &Event::FrontierSolved {
            round: 1,
            task: "not-actually-hard".into(),
            tokens_to_first_solve: 500,
            mutations_committed: 0,
        },
    )
    .unwrap();
    match l.by_kind("frontier_solved").unwrap()[0].event {
        Event::FrontierSolved {
            mutations_committed,
            ..
        } => assert_eq!(mutations_committed, 0),
        _ => panic!("wrong event"),
    }
}
