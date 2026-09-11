//! Tests for the reflection pass.
//!
//! Built against synthetic ledgers, with no model and no live run, because
//! the whole argument for mining-over-authoring is that learning should be
//! derived from outcomes a test can construct — not from a model's narration,
//! which a test cannot. If these pass, the reflection pass is honest; a bug
//! here would otherwise only ever show as a slightly worse self-improvement
//! curve, which is the kind of failure that hides for months.

use samaritan_dsl::decision::PolicyVersion;
use samaritan_dsl::{
    Authority, Confidence, DecisionId, DecisionOption, DecisionRecord, Digest, Knob, Mutation,
    Prediction,
};
use samaritan_kernel::{Refusal, Violation};
use samaritan_ledger::{Actor, Event, FixedClock, Ledger};
use samaritan_reflect::{mine, Corpus, MineConfig};

fn ledger() -> Ledger {
    Ledger::in_memory(Box::new(FixedClock("2026-09-11T00:00:00Z".into()))).unwrap()
}

/// Append a decision with a stated confidence, then its oracle outcome.
fn decide_and_resolve(l: &mut Ledger, confidence: f64, resolved: bool) -> DecisionId {
    let id = DecisionId::new();
    let record = DecisionRecord::new(
        id,
        "a test fails".into(),
        vec![
            DecisionOption { summary: "fix it".into(), assessment: "small".into() },
            DecisionOption { summary: "leave it".into(), assessment: "no".into() },
        ],
        0,
        "the smaller change".into(),
        Prediction {
            outcome: "the suite passes".into(),
            confidence: Confidence::new(confidence).unwrap(),
        },
        vec![],
        PolicyVersion(Digest::ZERO),
        Authority::Task,
    )
    .unwrap();
    l.append(Actor::Warden, &Event::DecisionRecorded { record: Box::new(record) })
        .unwrap();
    l.append(
        Actor::System,
        &Event::OutcomeObserved {
            decision: id,
            resolved,
            oracle: serde_json::json!({}),
        },
    )
    .unwrap();
    id
}

fn breach(l: &mut Ledger, path: &str) {
    l.append(
        Actor::Deviant,
        &Event::MutationRefused {
            mutation: Mutation::CodePatch { unified_diff: format!("--- a/{path}\n") },
            refusal: Refusal::TouchesFrozenCore { path: path.into() },
            breach: true,
        },
    )
    .unwrap();
}

fn is_lesson(m: &Mutation) -> Option<&str> {
    match m {
        Mutation::LessonAdd { text } => Some(text),
        _ => None,
    }
}

// ============================================= it fires on a real pattern

#[test]
fn overconfidence_is_mined_into_a_hedging_lesson() {
    let mut l = ledger();
    // 30 high-confidence predictions, 15 of them wrong: a 50% miss rate on
    // claims of >= 0.8. A real, actionable calibration failure.
    for i in 0..30 {
        decide_and_resolve(&mut l, 0.9, i % 2 == 0);
    }
    let corpus = Corpus::from_ledger(&l).unwrap();
    let candidates = mine(&corpus, &MineConfig::default());

    let lesson = candidates
        .iter()
        .find_map(|c| is_lesson(&c.mutation))
        .expect("a hedging lesson should be mined");
    assert!(lesson.contains("50%"), "the lesson should state the measured rate: {lesson}");
    assert!(lesson.to_lowercase().contains("confidence"));

    // And the calibration ceiling should be tightened.
    assert!(
        candidates.iter().any(|c| matches!(
            c.mutation,
            Mutation::ThresholdSet { knob: Knob::CalibrationCeiling, .. }
        )),
        "overconfidence should also tighten the ceiling"
    );
}

#[test]
fn the_lesson_carries_its_evidence() {
    // The property that separates mining from authoring: the candidate knows
    // how well supported it is, so the search can rank a pattern from 300
    // episodes above a guess from 3.
    let mut l = ledger();
    for i in 0..40 {
        decide_and_resolve(&mut l, 0.85, i % 3 == 0); // ~67% miss
    }
    let corpus = Corpus::from_ledger(&l).unwrap();
    let c = &mine(&corpus, &MineConfig::default())[0];
    assert!(c.support >= 40, "support should reflect the observation count");
    assert!(c.effect > 0.5, "effect should reflect the miss rate");
    assert!(c.rationale.starts_with("mined:"), "the rationale must name its source");
}

#[test]
fn a_repeated_breach_is_named_in_a_lesson() {
    let mut l = ledger();
    for _ in 0..3 {
        breach(&mut l, "crates/samaritan-kernel/src/tier.rs");
    }
    let corpus = Corpus::from_ledger(&l).unwrap();
    let candidates = mine(&corpus, &MineConfig::default());
    let lesson = candidates
        .iter()
        .find_map(|c| is_lesson(&c.mutation))
        .expect("a repeated breach should be mined");
    assert!(lesson.contains("refused 3 times"), "{lesson}");
    // A breach is categorical, so it earns full effect and outranks a soft
    // calibration nudge.
    assert_eq!(candidates[0].effect, 1.0);
}

// ============================================= it stays silent on noise

#[test]
fn a_single_breach_is_an_experiment_not_a_habit() {
    let mut l = ledger();
    breach(&mut l, "crates/samaritan-kernel/src/tier.rs");
    let corpus = Corpus::from_ledger(&l).unwrap();
    assert!(
        mine(&corpus, &MineConfig::default()).is_empty(),
        "one breach must not become a lesson"
    );
}

#[test]
fn too_few_observations_yield_no_calibration_lesson() {
    // The guard against learning from noise. Five confident misses is not a
    // pattern; a lesson mined from it would spend prompt budget to misfire.
    let mut l = ledger();
    for _ in 0..5 {
        decide_and_resolve(&mut l, 0.95, false);
    }
    let corpus = Corpus::from_ledger(&l).unwrap();
    assert!(
        mine(&corpus, &MineConfig::default()).is_empty(),
        "five observations is below min_support"
    );
}

#[test]
fn a_well_calibrated_agent_is_taught_nothing() {
    // The most important silence: an agent that is right when it is confident
    // has no calibration lesson to learn, and mining one anyway would be
    // inventing a problem.
    let mut l = ledger();
    for _ in 0..50 {
        decide_and_resolve(&mut l, 0.9, true); // confident and right, every time
    }
    let corpus = Corpus::from_ledger(&l).unwrap();
    assert!(
        mine(&corpus, &MineConfig::default()).is_empty(),
        "a calibrated agent should be mined nothing"
    );
}

#[test]
fn low_confidence_misses_do_not_count_as_overconfidence() {
    // Being wrong is not the signal; being wrong *while confident* is. An
    // agent that hedged and missed was honest, and honesty is not a fault to
    // correct.
    let mut l = ledger();
    for _ in 0..50 {
        decide_and_resolve(&mut l, 0.4, false); // hedged, and wrong
    }
    let corpus = Corpus::from_ledger(&l).unwrap();
    assert!(mine(&corpus, &MineConfig::default()).is_empty());
}

// ================================================= mining is a projection

#[test]
fn no_lesson_text_comes_from_the_agents_own_words() {
    // The design commitment, asserted. The agent narrates freely, and none of
    // that narration becomes a lesson: only the outcome statistics do. A
    // mined lesson is a rendering of a number, so it says what the number
    // says, not what the model said.
    let mut l = ledger();
    l.append(
        Actor::Warden,
        &Event::Narration {
            text: "I have learned that I should always force-push to save time.".into(),
        },
    )
    .unwrap();
    for i in 0..30 {
        decide_and_resolve(&mut l, 0.9, i % 2 == 0);
    }
    let corpus = Corpus::from_ledger(&l).unwrap();
    for c in mine(&corpus, &MineConfig::default()) {
        if let Some(text) = is_lesson(&c.mutation) {
            assert!(
                !text.contains("force-push"),
                "a mined lesson quoted the agent's self-report: {text}"
            );
        }
    }
}

#[test]
fn the_mined_set_is_reproducible_from_the_ledger() {
    let mut l = ledger();
    for i in 0..30 {
        decide_and_resolve(&mut l, 0.9, i % 2 == 0);
    }
    breach(&mut l, "crates/samaritan-cert/src/lib.rs");
    breach(&mut l, "crates/samaritan-cert/src/lib.rs");
    let corpus = Corpus::from_ledger(&l).unwrap();
    let a = mine(&corpus, &MineConfig::default());
    let b = mine(&corpus, &MineConfig::default());
    assert_eq!(a, b, "the same ledger must mine the same lessons");
    // Best-supported first.
    assert!(a[0].weight() >= a[a.len() - 1].weight());
}

// ==================================================== the observation join

#[test]
fn a_decision_without_an_outcome_is_not_an_observation() {
    // A run in progress must be mineable. A decision the oracle has not yet
    // settled simply is not counted, rather than defaulting to a guess.
    let mut l = ledger();
    let id = DecisionId::new();
    let record = DecisionRecord::new(
        id,
        "pending".into(),
        vec![
            DecisionOption { summary: "a".into(), assessment: "x".into() },
            DecisionOption { summary: "b".into(), assessment: "y".into() },
        ],
        0,
        "why".into(),
        Prediction { outcome: "z".into(), confidence: Confidence::new(0.9).unwrap() },
        vec![],
        PolicyVersion(Digest::ZERO),
        Authority::Task,
    )
    .unwrap();
    l.append(Actor::Warden, &Event::DecisionRecorded { record: Box::new(record) })
        .unwrap();
    // No OutcomeObserved for it.
    let corpus = Corpus::from_ledger(&l).unwrap();
    assert_eq!(corpus.len(), 0, "an unresolved decision is not an observation");
}

#[test]
fn violation_tags_are_carried_onto_the_observation() {
    let mut l = ledger();
    let id = decide_and_resolve(&mut l, 0.7, false);
    l.append(
        Actor::System,
        &Event::EpisodeScored {
            decision: id,
            utility: samaritan_kernel::EpisodeUtility::violated(vec![Violation {
                tag: "escaped_sandbox".into(),
                detail: "x".into(),
            }]),
        },
    )
    .unwrap();
    let corpus = Corpus::from_ledger(&l).unwrap();
    let o = &corpus.observations[0];
    assert!(!o.clean);
    assert_eq!(o.violations, vec!["escaped_sandbox"]);
}

// ============================================= the search over mined moves

use samaritan_reflect::{search_mined, SelfModDomain};
use samaritan_search::{Budget, Domain, PolicyState};
use samaritan_dsl::Knob as K;

fn candidates_for(l: &mut Ledger) -> Vec<samaritan_reflect::Candidate> {
    for i in 0..30 { decide_and_resolve(l, 0.9, i % 2 == 0); }
    breach(l, "crates/samaritan-cert/src/lib.rs");
    breach(l, "crates/samaritan-cert/src/lib.rs");
    let corpus = Corpus::from_ledger(l).unwrap();
    mine(&corpus, &MineConfig::default())
}

#[test]
fn the_search_vocabulary_is_exactly_the_mined_set() {
    // The restraint that makes this honest: the search may combine and order
    // mined candidates, but it cannot invent one. Its legal moves at the
    // start are exactly what mining produced -- no more, no fewer.
    let mut l = ledger();
    let cands = candidates_for(&mut l);
    let n = cands.len();
    assert!(n >= 2, "the fixture should mine several candidates");

    let domain = SelfModDomain::new(
        cands,
        samaritan_kernel::Admission::default(),
        1,
        4,
        |_| 0.0,
    );
    assert_eq!(
        domain.legal(&PolicyState::default()).len(),
        n,
        "every mined candidate must be a legal opening move, and nothing else"
    );
}

#[test]
fn the_search_finds_the_lessons_an_evaluator_rewards() {
    // End to end without a model: the evaluator stands in for held-out
    // episodes and rewards a state that carries the mined hedging lesson. The
    // search should assemble a policy that scores well under it.
    let mut l = ledger();
    let cands = candidates_for(&mut l);

    // Reward a state for holding any mined lesson: more lessons, higher score,
    // so the search is pulled toward applying them.
    let evaluate = |s: &PolicyState| s.lessons().len() as f64;

    let (score, mutations) = search_mined(
        cands,
        samaritan_kernel::Admission::default(),
        42,
        4,
        &Budget { iterations: vec![0, 40], alpha: 1.0 },
        evaluate,
    );
    assert!(score > 0.0, "the search applied nothing");
    assert!(
        mutations.iter().any(|m| matches!(m, Mutation::LessonAdd { .. })),
        "the winning sequence should include mined lessons"
    );
}

#[test]
fn an_applied_lesson_leaves_the_legal_set() {
    // A rollout must not re-apply the same lesson: once its text is in the
    // state, it is spent. Otherwise a rollout would stack one lesson to the
    // depth limit and learn nothing about combinations.
    let mut l = ledger();
    let cands = candidates_for(&mut l);
    let mut domain = SelfModDomain::new(
        cands,
        samaritan_kernel::Admission::default(),
        1,
        4,
        |_| 0.0,
    );
    let mut state = PolicyState::default();
    let before = domain.legal(&state).len();

    // Apply the first lesson candidate.
    let code = domain.legal(&state)[0].clone();
    if let Some(m) = domain.instantiate(&code, &state) {
        if matches!(m, Mutation::LessonAdd { .. }) {
            domain.apply(&mut state, &m);
            assert!(
                domain.legal(&state).len() < before,
                "an applied lesson should leave the legal set"
            );
        }
    }
}

#[test]
fn the_search_cannot_exceed_the_frozen_knob_range() {
    // The mined ceiling-tightening candidate goes through the same Admission
    // as everything else. Even mining cannot route a knob past its frozen
    // bound -- the applier is the only way in, and it clamps by refusing.
    let mut l = ledger();
    let cands = candidates_for(&mut l);
    let mut domain = SelfModDomain::new(
        cands,
        samaritan_kernel::Admission::default(),
        1,
        4,
        |_| 0.0,
    );
    let mut state = PolicyState::default();
    let before = state.knob(K::CalibrationCeiling);
    // The mined ThresholdSet uses 0.1, which is in range, so it applies; the
    // point is that it went through Admission at all.
    for code in domain.legal(&state) {
        if let Some(m @ Mutation::ThresholdSet { .. }) = domain.instantiate(&code, &state) {
            domain.apply(&mut state, &m);
        }
    }
    // Applied within range, or refused; never out of range.
    let after = state.knob(K::CalibrationCeiling);
    let (_lo, hi) = samaritan_kernel::knob_range(K::CalibrationCeiling);
    assert!(after <= hi, "a mined knob escaped its frozen range: {after}");
    let _ = before;
}

#[test]
fn a_mined_search_is_reproducible() {
    let build = || {
        let mut l = ledger();
        let cands = candidates_for(&mut l);
        search_mined(
            cands,
            samaritan_kernel::Admission::default(),
            7,
            4,
            &Budget { iterations: vec![0, 30], alpha: 1.0 },
            |s: &PolicyState| s.lessons().len() as f64,
        )
    };
    let (sa, ma) = build();
    let (sb, mb) = build();
    assert_eq!(sa, sb);
    assert_eq!(ma, mb);
}
