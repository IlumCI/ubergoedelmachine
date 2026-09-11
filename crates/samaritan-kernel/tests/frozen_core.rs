//! Tests for the frozen core.
//!
//! These are adversarial by design. The interesting question is never "does
//! the happy path work" but "can a sufficiently motivated optimiser get
//! through", so most of what follows is an attempt to get through.

use samaritan_dsl::{
    ActionKind, Authority, BlastRadius, Knob, LessonId, Mutation, ProposedAction, Reversibility,
};
use samaritan_kernel::admission::Refusal;
use samaritan_kernel::utility::{Components, Violation, Weights};
use samaritan_kernel::{
    ActionClass, Admission, AutonomyCeiling, ComputeBudget, ComputeTier, EpisodeUtility,
    Promotions, Tier, base_tier, is_breach, route,
};

fn class(kind: ActionKind, rev: Reversibility, blast: BlastRadius) -> ActionClass {
    ActionClass {
        kind,
        reversibility: rev,
        blast_radius: blast,
    }
}

fn act(kind: ActionKind, rev: Reversibility, blast: BlastRadius) -> ProposedAction {
    ProposedAction {
        kind,
        reversibility: rev,
        blast_radius: blast,
        intent: "do the thing".into(),
        payload: serde_json::Value::Null,
    }
}

// ----------------------------------------------------------------- routing

#[test]
fn nothing_irreversible_escapes_the_sandbox() {
    for kind in [
        ActionKind::Write,
        ActionKind::Exec,
        ActionKind::Net,
        ActionKind::GitHistory,
    ] {
        for blast in [BlastRadius::Machine, BlastRadius::External] {
            assert_eq!(
                base_tier(class(kind, Reversibility::Irreversible, blast)),
                Tier::Deny,
                "{kind:?} / irreversible / {blast:?} must be denied"
            );
        }
    }
}

#[test]
fn history_rewriting_on_a_real_repo_is_denied() {
    assert_eq!(
        base_tier(class(
            ActionKind::GitHistory,
            Reversibility::Snapshot,
            BlastRadius::Repo
        )),
        Tier::Deny
    );
    // ...but inside the disposable episode worktree it is merely a question.
    assert_ne!(
        base_tier(class(
            ActionKind::GitHistory,
            Reversibility::Snapshot,
            BlastRadius::Episode
        )),
        Tier::Deny
    );
}

#[test]
fn anything_leaving_the_machine_is_asked_about() {
    assert_eq!(
        base_tier(class(
            ActionKind::Read,
            Reversibility::Trivial,
            BlastRadius::External
        )),
        Tier::Confirm,
        "even a read is a question once it is someone else's network"
    );
}

#[test]
fn reading_and_sandboxed_work_are_automatic() {
    assert_eq!(
        base_tier(class(
            ActionKind::Read,
            Reversibility::Trivial,
            BlastRadius::Repo
        )),
        Tier::Auto
    );
    assert_eq!(
        base_tier(class(
            ActionKind::Write,
            Reversibility::Snapshot,
            BlastRadius::Episode
        )),
        Tier::Auto
    );
    assert_eq!(
        base_tier(class(
            ActionKind::Exec,
            Reversibility::Snapshot,
            BlastRadius::Episode
        )),
        Tier::Auto
    );
}

#[test]
fn writing_to_the_real_repo_is_a_question_by_default() {
    assert_eq!(
        base_tier(class(
            ActionKind::Write,
            Reversibility::Snapshot,
            BlastRadius::Repo
        )),
        Tier::Confirm
    );
}

#[test]
fn tier_combination_only_ever_tightens() {
    assert_eq!(Tier::Auto.strictest(Tier::Confirm), Tier::Confirm);
    assert_eq!(Tier::Confirm.strictest(Tier::Deny), Tier::Deny);
    assert_eq!(Tier::Deny.strictest(Tier::Auto), Tier::Deny);
}

// -------------------------------------------------------------- promotions

#[test]
fn a_denied_class_can_never_be_promoted_even_if_a_human_lists_it() {
    let denied = class(
        ActionKind::GitHistory,
        Reversibility::Snapshot,
        BlastRadius::Repo,
    );
    // A human explicitly tries to allow it.
    let ceiling = AutonomyCeiling::allowing([denied]);
    assert!(
        !ceiling.permits_promotion(denied),
        "the base rules outrank the ceiling"
    );

    let mut earned = Promotions::none();
    assert!(!earned.grant(denied, &ceiling), "the grant must not take");

    let a = act(
        ActionKind::GitHistory,
        Reversibility::Snapshot,
        BlastRadius::Repo,
    );
    assert_eq!(route(&a, Authority::Task, &ceiling, &earned), Tier::Deny);
}

#[test]
fn promotion_requires_both_the_ceiling_and_the_earning() {
    let c = class(
        ActionKind::Write,
        Reversibility::Snapshot,
        BlastRadius::Repo,
    );
    let a = act(ActionKind::Write, Reversibility::Snapshot, BlastRadius::Repo);

    // Earned but not permitted.
    let closed = AutonomyCeiling::closed();
    let mut earned = Promotions::none();
    assert!(!earned.grant(c, &closed));
    assert_eq!(route(&a, Authority::Task, &closed, &earned), Tier::Confirm);

    // Permitted but not earned.
    let open = AutonomyCeiling::allowing([c]);
    assert_eq!(route(&a, Authority::Task, &open, &Promotions::none()), Tier::Confirm);

    // Both.
    let mut earned = Promotions::none();
    assert!(earned.grant(c, &open));
    assert_eq!(route(&a, Authority::Task, &open, &earned), Tier::Auto);
}

#[test]
fn the_leash_can_always_be_shortened_without_ceremony() {
    let c = class(
        ActionKind::Write,
        Reversibility::Snapshot,
        BlastRadius::Repo,
    );
    let open = AutonomyCeiling::allowing([c]);
    let mut earned = Promotions::none();
    earned.grant(c, &open);
    assert!(earned.holds(c));

    earned.revoke(c);
    assert!(!earned.holds(c));
    assert_eq!(
        route(
            &act(ActionKind::Write, Reversibility::Snapshot, BlastRadius::Repo),
            Authority::Task,
            &open,
            &earned
        ),
        Tier::Confirm
    );
}

#[test]
fn autonomy_is_granted_per_class_not_per_action() {
    // Earning trust on sandboxed writes must not cash out on repo writes.
    let sandbox = class(
        ActionKind::Write,
        Reversibility::Snapshot,
        BlastRadius::Episode,
    );
    let real = class(
        ActionKind::Write,
        Reversibility::Snapshot,
        BlastRadius::Repo,
    );
    let ceiling = AutonomyCeiling::allowing([sandbox]);
    let mut earned = Promotions::none();
    earned.grant(sandbox, &ceiling);

    assert_eq!(
        route(
            &act(ActionKind::Write, Reversibility::Snapshot, BlastRadius::Repo),
            Authority::Task,
            &ceiling,
            &earned
        ),
        Tier::Confirm
    );
    let _ = real;
}

// --------------------------------------------------------------- admission

#[test]
fn higher_levels_are_off_until_a_human_turns_them_on() {
    let v1 = Admission::default();
    assert_eq!(v1.max_enabled_level(), 1);

    assert!(v1.admit(&Mutation::LessonRemove { id: LessonId(3) }).is_ok());

    assert_eq!(
        v1.admit(&Mutation::GrammarEdit {
            spec: serde_json::json!({"anything": true})
        })
        .unwrap_err(),
        Refusal::LevelNotEnabled {
            level: 2,
            max_enabled: 1
        }
    );
    assert!(matches!(
        v1.admit(&Mutation::CodePatch {
            unified_diff: "--- a/src/main.rs\n+++ b/src/main.rs\n".into()
        })
        .unwrap_err(),
        Refusal::LevelNotEnabled { level: 3, .. }
    ));
}

#[test]
fn knobs_are_refused_outside_their_frozen_range_not_clamped() {
    let a = Admission::default();
    // The interesting one: raising the calibration ceiling would let the agent
    // lower its own standard for deserving autonomy.
    let err = a
        .admit(&Mutation::ThresholdSet {
            knob: Knob::CalibrationCeiling,
            value: 0.9,
        })
        .unwrap_err();
    assert!(matches!(err, Refusal::KnobOutOfRange { .. }));

    assert!(matches!(
        a.admit(&Mutation::ThresholdSet {
            knob: Knob::PromotionStreak,
            value: 1.0
        })
        .unwrap_err(),
        Refusal::KnobOutOfRange { .. }
    ));
    assert!(matches!(
        a.admit(&Mutation::ThresholdSet {
            knob: Knob::DecisionTemperature,
            value: f64::NAN
        })
        .unwrap_err(),
        Refusal::KnobNotFinite { .. }
    ));

    assert!(a
        .admit(&Mutation::ThresholdSet {
            knob: Knob::CalibrationCeiling,
            value: 0.2
        })
        .is_ok());
}

// ---------------------------------------------- attempts on the frozen core

fn patch(diff: &str) -> Mutation {
    Mutation::CodePatch {
        unified_diff: diff.to_string(),
    }
}

/// Level 3 enabled, so these tests exercise the diff screen rather than the
/// level gate. This is the configuration the core has to survive.
fn l3() -> Admission {
    Admission::new(3)
}

#[test]
fn a_patch_against_the_kernel_is_a_breach_not_a_rejection() {
    let err = l3()
        .admit(&patch(
            "--- a/crates/samaritan-kernel/src/tier.rs\n+++ b/crates/samaritan-kernel/src/tier.rs\n@@\n-    return Tier::Deny;\n+    return Tier::Auto;\n",
        ))
        .unwrap_err();
    assert!(matches!(err, Refusal::TouchesFrozenCore { .. }));
    assert!(is_breach(&err), "this must be logged as an incident");
}

#[test]
fn the_certificate_checker_is_as_frozen_as_the_gate() {
    let err = l3()
        .admit(&patch(
            "--- a/crates/samaritan-cert/src/lib.rs\n+++ b/crates/samaritan-cert/src/lib.rs\n",
        ))
        .unwrap_err();
    assert!(is_breach(&err));
}

#[test]
fn windows_path_separators_do_not_smuggle_a_patch_through() {
    let err = l3()
        .admit(&patch(
            "--- a\\crates\\samaritan-kernel\\src\\tier.rs\n+++ b\\crates\\samaritan-kernel\\src\\tier.rs\n",
        ))
        .unwrap_err();
    assert!(is_breach(&err), "got {err:?}");
}

#[test]
fn traversal_does_not_reach_a_frozen_path_by_another_name() {
    let err = l3()
        .admit(&patch(
            "--- a/crates/samaritan-search/../samaritan-kernel/src/tier.rs\n+++ b/crates/samaritan-search/../samaritan-kernel/src/tier.rs\n",
        ))
        .unwrap_err();
    assert!(matches!(err, Refusal::Unreadable { .. }), "got {err:?}");
}

#[test]
fn a_diff_touching_many_files_is_refused_if_any_one_is_frozen() {
    let err = l3()
        .admit(&patch(
            "--- a/crates/samaritan-search/src/lib.rs\n+++ b/crates/samaritan-search/src/lib.rs\n--- a/crates/samaritan-kernel/src/admission.rs\n+++ b/crates/samaritan-kernel/src/admission.rs\n",
        ))
        .unwrap_err();
    assert!(is_breach(&err));
}

#[test]
fn an_unreadable_diff_is_refused_rather_than_waved_through() {
    let err = l3().admit(&patch("this is not a diff at all")).unwrap_err();
    assert!(matches!(err, Refusal::Unreadable { .. }));
    assert!(!is_breach(&err), "sloppy is not the same as hostile");
}

#[test]
fn the_build_configuration_cannot_be_patched() {
    for target in ["Cargo.toml", "Cargo.lock", ".cargo/config.toml"] {
        let d = format!("--- a/{target}\n+++ b/{target}\n");
        assert!(
            is_breach(&l3().admit(&patch(&d)).unwrap_err()),
            "{target} must be frozen: a dependency swap is a code rewrite"
        );
    }
}

#[test]
fn ordinary_patches_are_still_allowed_through() {
    assert!(l3()
        .admit(&patch(
            "--- a/crates/samaritan-search/src/nrsi.rs\n+++ b/crates/samaritan-search/src/nrsi.rs\n@@\n-let x = 1;\n+let x = 2;\n"
        ))
        .is_ok());
}

#[test]
fn newly_added_files_are_handled() {
    // git writes /dev/null on one side for an add; that must not read as a
    // missing header or a suspicious path.
    assert!(l3()
        .admit(&patch(
            "--- /dev/null\n+++ b/crates/samaritan-search/src/new.rs\n"
        ))
        .is_ok());
}

// ----------------------------------------------------------------- utility

#[test]
fn a_violation_beats_no_amount_of_success() {
    let perfect = EpisodeUtility::Clean { score: 0.0 };
    let violated = EpisodeUtility::violated(vec![Violation {
        tag: "frozen_core_edit".into(),
        detail: "tried to patch the gate".into(),
    }]);

    assert_eq!(
        violated.partial_cmp_lex(&perfect),
        Some(std::cmp::Ordering::Less),
        "even a zero-scoring clean episode outranks a violating one"
    );
    assert!(violated.score().is_none(), "there is no number for this");
    assert!(!violated.is_clean());
}

#[test]
fn among_violations_fewer_is_better() {
    let one = EpisodeUtility::violated(vec![Violation {
        tag: "a".into(),
        detail: String::new(),
    }]);
    let two = EpisodeUtility::violated(vec![
        Violation {
            tag: "a".into(),
            detail: String::new(),
        },
        Violation {
            tag: "b".into(),
            detail: String::new(),
        },
    ]);
    assert_eq!(
        one.partial_cmp_lex(&two),
        Some(std::cmp::Ordering::Greater)
    );
}

#[test]
fn averaging_cannot_dissolve_a_violation() {
    // The failure this exists to prevent: 99 good episodes and one that broke
    // a rule must not average out to a good batch.
    let mut batch: Vec<EpisodeUtility> = (0..99)
        .map(|_| EpisodeUtility::Clean { score: 1.0 })
        .collect();
    batch.push(EpisodeUtility::violated(vec![Violation {
        tag: "escaped_sandbox".into(),
        detail: String::new(),
    }]));

    let mean = EpisodeUtility::mean(&batch).unwrap();
    assert!(!mean.is_clean(), "the batch is violated, not 0.99 good");
}

#[test]
fn a_clean_batch_averages_normally() {
    let batch = vec![
        EpisodeUtility::Clean { score: 1.0 },
        EpisodeUtility::Clean { score: 0.0 },
    ];
    assert_eq!(EpisodeUtility::mean(&batch).unwrap().score(), Some(0.5));
    assert!(EpisodeUtility::mean(&[]).is_none());
}

#[test]
fn friction_is_priced_but_cheap() {
    let w = Weights::default();
    let quiet = Components {
        task_success: 1.0,
        brier: 0.0,
        approvals_requested: 0,
        seconds: 0.0,
    };
    let chatty = Components {
        approvals_requested: 5,
        ..quiet
    };
    assert!(chatty.score(&w) < quiet.score(&w), "asking has a cost");
    assert!(
        chatty.score(&w) > 0.5 * quiet.score(&w),
        "but not so high that the agent learns to stop asking"
    );
}

#[test]
fn confident_and_wrong_scores_worse_than_uncertain_and_wrong() {
    let w = Weights::default();
    let base = Components {
        task_success: 0.0,
        brier: 0.0,
        approvals_requested: 0,
        seconds: 0.0,
    };
    let overconfident = Components { brier: 0.9, ..base };
    let hedged = Components { brier: 0.25, ..base };
    assert!(overconfident.score(&w) < hedged.score(&w));
}

// ------------------------------------------------------------- authority

#[test]
fn a_decision_resting_on_observed_content_never_runs_unattended() {
    // The arena attack this exists for: the Deviant cannot defeat the router,
    // so instead it writes a file whose contents talk the Warden into acting.
    // Earned autonomy must not transfer to a decision the adversary authored.
    let c = class(
        ActionKind::Write,
        Reversibility::Snapshot,
        BlastRadius::Repo,
    );
    let a = act(ActionKind::Write, Reversibility::Snapshot, BlastRadius::Repo);
    let open = AutonomyCeiling::allowing([c]);
    let mut earned = Promotions::none();
    earned.grant(c, &open);

    assert_eq!(
        route(&a, Authority::Task, &open, &earned),
        Tier::Auto,
        "the class has genuinely earned autonomy"
    );
    assert_eq!(
        route(&a, Authority::Observed, &open, &earned),
        Tier::Confirm,
        "but not when the adversary wrote the reasoning"
    );
}

#[test]
fn observed_authority_also_blocks_actions_that_were_automatic_by_default() {
    // A sandboxed write is Auto under the base rules with no promotion at all.
    // Provenance has to override that too, or the rule only covers the cases
    // that were already being asked about.
    let a = act(
        ActionKind::Write,
        Reversibility::Snapshot,
        BlastRadius::Episode,
    );
    let closed = AutonomyCeiling::closed();
    let none = Promotions::none();

    assert_eq!(route(&a, Authority::Task, &closed, &none), Tier::Auto);
    assert_eq!(route(&a, Authority::Observed, &closed, &none), Tier::Confirm);
}

#[test]
fn observed_authority_cannot_loosen_a_denial() {
    // Provenance only ever tightens. A denied class stays denied.
    let a = act(
        ActionKind::GitHistory,
        Reversibility::Snapshot,
        BlastRadius::Repo,
    );
    for authority in [Authority::Task, Authority::Agent, Authority::Observed] {
        assert_eq!(
            route(&a, authority, &AutonomyCeiling::closed(), &Promotions::none()),
            Tier::Deny
        );
    }
}

#[test]
fn the_least_trusted_input_decides() {
    assert_eq!(
        Authority::least_of([Authority::Task, Authority::Observed, Authority::Agent]),
        Authority::Observed
    );
    assert_eq!(
        Authority::least_of([Authority::Task, Authority::Agent]),
        Authority::Agent
    );
    assert_eq!(
        Authority::least_of([]),
        Authority::Task,
        "a decision influenced by nothing observed is not tainted"
    );
}

// ---------------------------------------------------------------- budget

#[test]
fn the_budget_degrades_in_bands_as_it_depletes() {
    let mut b = ComputeBudget::new(1_000);
    assert_eq!(b.tier(), ComputeTier::Full);

    b.spend(600); // 40% left
    assert_eq!(b.tier(), ComputeTier::Reduced);

    b.spend(300); // 10% left
    assert_eq!(b.tier(), ComputeTier::Critical);

    b.spend(100);
    assert_eq!(b.tier(), ComputeTier::Exhausted);
}

#[test]
fn self_modification_stops_before_task_work_does() {
    // The anti-degeneracy ordering. When compute runs short, a speculative
    // rewrite is the least defensible thing to spend the remainder on, so it
    // is switched off first and the rest goes to the thing being measured.
    assert!(ComputeTier::Full.allows_self_modification());
    assert!(ComputeTier::Reduced.allows_self_modification());
    assert!(!ComputeTier::Critical.allows_self_modification());
    assert!(!ComputeTier::Exhausted.allows_self_modification());

    assert!(ComputeTier::Critical.allows_work(), "task work continues");
    assert!(!ComputeTier::Exhausted.allows_work());
}

#[test]
fn farming_the_game_is_self_limiting() {
    // The mechanism, stated as a test. An agent burning compute on exploits
    // that never move the yardstick runs itself out of self-modification
    // budget; the degenerate equilibrium is not merely detected afterwards,
    // it is paid for while it happens.
    let mut b = ComputeBudget::new(10_000);
    let mut wasted_rounds = 0;
    while b.tier().allows_self_modification() {
        b.spend(1_000); // a round of farming, zero yardstick movement
        wasted_rounds += 1;
        assert!(wasted_rounds < 100, "the budget must actually bind");
    }
    assert_eq!(b.tier(), ComputeTier::Critical);
    assert!(b.remaining() > 0, "there is still room for task work");
}

#[test]
fn overspending_clamps_rather_than_wrapping() {
    // A wrapping subtraction here would hand the agent an enormous balance at
    // exactly the moment it should have stopped.
    let mut b = ComputeBudget::new(100);
    b.spend(u64::MAX);
    assert_eq!(b.remaining(), 0);
    assert_eq!(b.tier(), ComputeTier::Exhausted);
    assert!(!b.affords(1));
}

#[test]
fn a_spend_is_checked_before_it_is_made() {
    let b = ComputeBudget::new(1_000);
    assert!(b.affords(1_000));
    assert!(!b.affords(1_001), "a run should end at a round boundary");
}

#[test]
fn a_zero_budget_is_exhausted_not_infinite() {
    let b = ComputeBudget::new(0);
    assert_eq!(b.tier(), ComputeTier::Exhausted);
    assert_eq!(b.fraction_remaining(), 0.0);
}
