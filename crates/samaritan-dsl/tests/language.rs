//! Tests for the decision language.
//!
//! The theme throughout: records arrive from a language model, so the tests
//! that matter most are the ones proving malformed input is rejected at the
//! deserialization boundary rather than only by the constructor a caller might
//! forget to use.

use samaritan_dsl::decision::PolicyVersion;
use samaritan_dsl::mutation::{Knob, LessonId};
use samaritan_dsl::{
    ActionKind, BlastRadius, Confidence, DecisionId, DecisionOption, DecisionRecord, Digest,
    DslError, Mutation, MutationCode, MutationPolicy, Prediction, ProposedAction, Reversibility,
};

fn opt(s: &str) -> DecisionOption {
    DecisionOption {
        summary: s.to_string(),
        assessment: "considered".to_string(),
    }
}

fn prediction() -> Prediction {
    Prediction {
        outcome: "tests go green".to_string(),
        confidence: Confidence::new(0.7).unwrap(),
    }
}

fn action() -> ProposedAction {
    ProposedAction {
        kind: ActionKind::Write,
        reversibility: Reversibility::Snapshot,
        blast_radius: BlastRadius::Episode,
        intent: "patch the failing assertion".to_string(),
        payload: serde_json::json!({"path": "src/lib.rs"}),
    }
}

fn record_with(options: Vec<DecisionOption>, chosen: usize) -> Result<DecisionRecord, DslError> {
    DecisionRecord::new(
        DecisionId::new(),
        "two tests fail after the refactor".to_string(),
        options,
        chosen,
        "the smaller patch is easier to undo".to_string(),
        prediction(),
        vec![action()],
        PolicyVersion(Digest::ZERO),
        samaritan_dsl::Authority::Task,
    )
}

// ---------------------------------------------------------------- validation

#[test]
fn a_single_option_is_not_a_decision() {
    let err = record_with(vec![opt("just do it")], 0).unwrap_err();
    assert_eq!(err, DslError::TooFewOptions(1));
}

#[test]
fn chosen_must_index_an_option_that_exists() {
    let err = record_with(vec![opt("a"), opt("b")], 2).unwrap_err();
    assert_eq!(
        err,
        DslError::ChosenOutOfRange {
            chosen: 2,
            count: 2
        }
    );
}

#[test]
fn blank_text_is_rejected_including_whitespace_only() {
    let err = record_with(vec![opt("a"), opt("   ")], 0).unwrap_err();
    assert_eq!(
        err,
        DslError::Blank {
            field: "option.summary"
        }
    );
}

#[test]
fn a_valid_record_exposes_the_option_it_chose() {
    let r = record_with(vec![opt("small patch"), opt("rewrite the module")], 0).unwrap();
    assert_eq!(r.chosen().summary, "small patch");
    assert_eq!(r.chosen_index(), 0);
    assert_eq!(r.actions().len(), 1);
}

#[test]
fn confidence_accepts_the_closed_unit_interval_and_nothing_else() {
    assert!(Confidence::new(0.0).is_ok());
    assert!(Confidence::new(1.0).is_ok());
    assert_eq!(
        Confidence::new(1.5).unwrap_err(),
        DslError::ConfidenceOutOfRange(1.5)
    );
    assert_eq!(
        Confidence::new(-0.001).unwrap_err(),
        DslError::ConfidenceOutOfRange(-0.001)
    );
    assert!(matches!(
        Confidence::new(f64::NAN).unwrap_err(),
        DslError::ConfidenceNotFinite(_)
    ));
    assert!(matches!(
        Confidence::new(f64::INFINITY).unwrap_err(),
        DslError::ConfidenceNotFinite(_)
    ));
}

// ------------------------------------------------- the boundary that matters

#[test]
fn round_trips_through_json() {
    let r = record_with(vec![opt("a"), opt("b")], 1).unwrap();
    let json = serde_json::to_string(&r).unwrap();
    let back: DecisionRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(r, back);
}

#[test]
fn deserialization_enforces_the_same_invariants_as_the_constructor() {
    // A model could emit this. The type must not be constructible from it.
    let json = serde_json::json!({
        "id": "6f8b4c1e-0000-4000-8000-000000000000",
        "situation": "something broke",
        "options": [{"summary": "only one", "assessment": "the only way"}],
        "chosen": 0,
        "rationale": "obvious",
        "prediction": {"outcome": "fixed", "confidence": 0.9},
        "actions": [],
        "policy_version": "0000000000000000000000000000000000000000000000000000000000000000"
    });
    let err = serde_json::from_value::<DecisionRecord>(json).unwrap_err();
    assert!(
        err.to_string().contains("at least two options"),
        "expected the arity rule to fire, got: {err}"
    );
}

#[test]
fn deserialization_rejects_out_of_range_confidence() {
    let json = serde_json::json!({
        "id": "6f8b4c1e-0000-4000-8000-000000000000",
        "situation": "something broke",
        "options": [
            {"summary": "a", "assessment": "x"},
            {"summary": "b", "assessment": "y"}
        ],
        "chosen": 0,
        "rationale": "obvious",
        "prediction": {"outcome": "fixed", "confidence": 4.2},
        "actions": [],
        "policy_version": "0000000000000000000000000000000000000000000000000000000000000000"
    });
    assert!(serde_json::from_value::<DecisionRecord>(json).is_err());
}

// --------------------------------------------------------------------- hash

#[test]
fn digests_are_stable_and_chain_order_matters() {
    let a = Digest::of_bytes(b"alpha");
    assert_eq!(a, Digest::of_bytes(b"alpha"));
    assert_ne!(a, Digest::of_bytes(b"beta"));

    let b = Digest::of_bytes(b"beta");
    assert_ne!(
        Digest::chain(&a, b"beta"),
        Digest::chain(&b, b"alpha"),
        "chaining must not be commutative or the ledger can be reordered"
    );
}

#[test]
fn digest_hex_round_trips() {
    let d = Digest::of_bytes(b"samaritan");
    assert_eq!(Digest::from_hex(&d.to_hex()), Some(d));
    assert_eq!(Digest::from_hex("nonsense"), None);
    assert_eq!(d.short().len(), 12);
}

// ----------------------------------------------------------------- mutation

#[test]
fn mutation_codes_group_by_class_not_by_payload() {
    let a = Mutation::LessonAdd {
        text: "always read the test before patching".into(),
    };
    let b = Mutation::LessonAdd {
        text: "something else entirely".into(),
    };
    assert_eq!(a.code(), b.code(), "the policy generalises over classes");

    let t1 = Mutation::ThresholdSet {
        knob: Knob::PromotionStreak,
        value: 5.0,
    };
    let t2 = Mutation::ThresholdSet {
        knob: Knob::LessonBudget,
        value: 8.0,
    };
    assert_ne!(t1.code(), t2.code(), "but distinct knobs are distinct moves");
}

#[test]
fn mutations_declare_the_level_they_belong_to() {
    assert_eq!(Mutation::LessonRemove { id: LessonId(1) }.level(), 1);
    assert_eq!(
        Mutation::LessonReweight {
            id: LessonId(1),
            delta: 0.5
        }
        .level(),
        1
    );
    assert_eq!(
        Mutation::GrammarEdit {
            spec: serde_json::json!({})
        }
        .level(),
        2
    );
    assert_eq!(
        Mutation::CodePatch {
            unified_diff: String::new()
        }
        .level(),
        3
    );
}

// ------------------------------------------------------------------- policy

fn codes(names: &[&str]) -> Vec<MutationCode> {
    names.iter().map(|n| MutationCode(n.to_string())).collect()
}

#[test]
fn an_untouched_policy_is_uniform() {
    let p = MutationPolicy::new();
    let legal = codes(&["a", "b", "c", "d"]);
    for pr in p.distribution(&legal) {
        assert!((pr - 0.25).abs() < 1e-12);
    }
}

#[test]
fn distribution_sums_to_one_and_survives_extreme_weights() {
    let mut p = MutationPolicy::new();
    p.set_weight(MutationCode("a".into()), 800.0).unwrap();
    p.set_weight(MutationCode("b".into()), -800.0).unwrap();
    let legal = codes(&["a", "b"]);
    let d = p.distribution(&legal);
    let sum: f64 = d.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "softmax must not overflow");
    assert!(d[0] > 0.99);
}

#[test]
fn distribution_of_nothing_is_empty_rather_than_a_panic() {
    assert!(MutationPolicy::new().distribution(&[]).is_empty());
}

#[test]
fn adapt_moves_weight_toward_the_chosen_move() {
    let p = MutationPolicy::new();
    let legal = codes(&["a", "b", "c", "d"]);
    let trace = vec![(MutationCode("a".into()), legal.clone())];

    let q = p.adapt(&trace, 1.0);
    assert!(q.weight(&MutationCode("a".into())) > p.weight(&MutationCode("a".into())));
    for c in ["b", "c", "d"] {
        assert!(q.weight(&MutationCode(c.into())) < 0.0);
    }

    // Repeated adaptation toward the same move must keep raising its share.
    let before = q.distribution(&legal)[0];
    let r = q.adapt(&trace, 1.0);
    assert!(r.distribution(&legal)[0] > before);
}

#[test]
fn adapt_is_a_zero_sum_reallocation() {
    // Each step adds alpha to the chosen move and removes alpha spread across
    // the legal set, so total weight is conserved. If this drifts, the softmax
    // slowly saturates for reasons unrelated to the search.
    let p = MutationPolicy::new();
    let legal = codes(&["a", "b", "c"]);
    let trace = vec![
        (MutationCode("a".into()), legal.clone()),
        (MutationCode("c".into()), legal.clone()),
    ];
    let q = p.adapt(&trace, 0.7);
    let total: f64 = q.iter().map(|(_, w)| w).sum();
    assert!(total.abs() < 1e-12, "weights drifted by {total}");
}

#[test]
fn adapt_is_out_of_place() {
    // Rosin computes the softmax normaliser from the *old* policy while updates
    // accumulate into the copy. A caller must be able to keep the original.
    let p = MutationPolicy::new();
    let legal = codes(&["a", "b"]);
    let _ = p.adapt(&[(MutationCode("a".into()), legal)], 1.0);
    assert!(p.is_empty(), "adapt must not touch its receiver");
}

#[test]
fn clone_is_deeply_independent() {
    // The property NRPA's rollback depends on. If MutationPolicy ever gains
    // shared mutable state, adaptations made inside a deeper recursion level
    // would leak back into the caller and the isolation between meta-levels
    // would be silently gone.
    let mut original = MutationPolicy::new();
    original.set_weight(MutationCode("a".into()), 1.0).unwrap();

    let mut copy = original.clone();
    copy.set_weight(MutationCode("a".into()), 99.0).unwrap();
    copy.set_weight(MutationCode("b".into()), 5.0).unwrap();

    assert_eq!(original.weight(&MutationCode("a".into())), 1.0);
    assert_eq!(original.weight(&MutationCode("b".into())), 0.0);
    assert_eq!(original.len(), 1);
}

#[test]
fn non_finite_weights_are_refused() {
    let mut p = MutationPolicy::new();
    assert!(matches!(
        p.set_weight(MutationCode("a".into()), f64::NAN),
        Err(DslError::BadWeight(_))
    ));
}
