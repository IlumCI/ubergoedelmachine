//! Level 2 as an actual search: looking for a better move generator.
//!
//! Making the grammar editable was necessary and not sufficient — something has
//! to *search* over it. This is that something, and the construction follows the
//! design's definition of the nesting axis exactly:
//!
//! > a rollout at level 2 **is** a level-1 search.
//!
//! So a [`GrammarDomain`]'s moves are grammar edits, and the score of a grammar
//! is whatever a level-1 search achieves *while running under it*. That is what
//! makes the level meaningful rather than decorative: a grammar is not judged by
//! how it looks, it is judged by how well the search below it does when it has
//! to use it.
//!
//! The cost structure falls out and explains the cadence. Every level-2 rollout
//! pays for a whole level-1 search, so level 2 is a nightly event where level 1
//! is continuous — the product of the iteration counts, exactly as the budget
//! comment says.
//!
//! The inner search is injected as a closure, so the level-2 wiring is testable
//! without episodes, a model, or a level-1 run — the same seam
//! [`crate::SelfModDomain`] uses for its evaluator, and for the same reason.

use samaritan_dsl::{Mutation, MutationCode, MutationPolicy};
use samaritan_kernel::Admission;
use samaritan_search::{Budget, Domain, GrammarOp, PolicyState, nrsi};

/// Scores a grammar by running the search that has to live with it.
pub type InnerSearch<'a> = dyn FnMut(&PolicyState) -> f64 + 'a;

/// The default vocabulary of grammar edits.
///
/// Deliberately a small, bounded sweep rather than an open space: the point of
/// level 2 is to find a better move generator, not to wander. Every op here is
/// one the grammar's own invariants will accept or reject on its merits — none
/// of them can promote the search a level, because the enableable-kind
/// whitelist in `samaritan-search` does not contain `code_patch`.
pub fn grammar_ops() -> Vec<GrammarOp> {
    let mut ops = Vec::new();
    for value in [0.25, 0.5, 1.0, 2.0] {
        ops.push(GrammarOp::SetReweightDelta { value });
    }
    for value in [2, 4, 8] {
        ops.push(GrammarOp::SetDepth { value });
    }
    for value in [0.5, 1.0, 2.0] {
        ops.push(GrammarOp::SetAdaptAlpha { value });
    }
    // Narrowing the vocabulary is a real move: a grammar that stops offering a
    // move the evidence does not support is a better grammar.
    for kind in ["lesson_remove", "lesson_reweight", "threshold_set"] {
        ops.push(GrammarOp::DisableKind { kind: kind.to_string() });
        ops.push(GrammarOp::EnableKind { kind: kind.to_string() });
    }
    ops
}

/// A [`Domain`] whose moves are grammar edits.
pub struct GrammarDomain<'a> {
    ops: Vec<(MutationCode, GrammarOp)>,
    admission: Admission,
    inner: Box<InnerSearch<'a>>,
    rng: u64,
    depth: usize,
}

impl<'a> GrammarDomain<'a> {
    /// `inner` must run a level-1 search from the given state and return its
    /// score — that is what makes the level-2 rollout a level-1 search.
    pub fn new(
        ops: Vec<GrammarOp>,
        admission: Admission,
        seed: u64,
        depth: usize,
        inner: impl FnMut(&PolicyState) -> f64 + 'a,
    ) -> Self {
        let ops = ops
            .into_iter()
            .enumerate()
            .map(|(i, op)| (MutationCode(format!("gram:{i}:{}", op_name(&op))), op))
            .collect();
        Self {
            ops,
            admission,
            inner: Box::new(inner),
            rng: seed.max(1),
            depth,
        }
    }

    fn next_unit(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn op_name(op: &GrammarOp) -> &'static str {
    match op {
        GrammarOp::EnableKind { .. } => "enable",
        GrammarOp::DisableKind { .. } => "disable",
        GrammarOp::SetReweightDelta { .. } => "reweight",
        GrammarOp::SetDepth { .. } => "depth",
        GrammarOp::SetAdaptAlpha { .. } => "alpha",
    }
}

impl Domain for GrammarDomain<'_> {
    fn legal(&self, _state: &PolicyState) -> Vec<MutationCode> {
        self.ops.iter().map(|(c, _)| c.clone()).collect()
    }

    fn instantiate(&mut self, code: &MutationCode, _state: &PolicyState) -> Option<Mutation> {
        let (_, op) = self.ops.iter().find(|(c, _)| c == code)?;
        Some(Mutation::GrammarEdit {
            spec: serde_json::to_value(op).ok()?,
        })
    }

    fn apply(&mut self, state: &mut PolicyState, m: &Mutation) -> bool {
        // A refused grammar edit — one that would empty the vocabulary, or set a
        // nonsense parameter — is simply a dead move. The search learns it does
        // not pay rather than being stopped by it.
        state.apply(m, &self.admission).is_ok()
    }

    fn evaluate(&mut self, state: &PolicyState) -> f64 {
        (self.inner)(state)
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn random(&mut self) -> f64 {
        self.next_unit()
    }
}

/// Search for a better grammar, scoring each by the level-1 search it produces.
///
/// Returns the winning grammar edits. Like every other search here, this only
/// finds a *candidate*: whether it may be committed is `samaritan-cert`'s call,
/// on held-out tasks, because the score below was computed on the very data the
/// search optimised against.
pub fn search_grammar<'a>(
    admission: Admission,
    seed: u64,
    depth: usize,
    budget: &Budget,
    inner: impl FnMut(&PolicyState) -> f64 + 'a,
) -> (f64, Vec<Mutation>) {
    let mut domain = GrammarDomain::new(grammar_ops(), admission, seed, depth, inner);
    let base = PolicyState::default();
    // Level 2: the rollouts below are level-1 searches, supplied by `inner`.
    let (score, trace) = nrsi(2, &MutationPolicy::new(), &mut domain, &base, budget);
    (score, trace.mutations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use samaritan_search::Grammar;

    fn budget() -> Budget {
        Budget {
            iterations: vec![0, 3, 3, 1],
            alpha: 1.0,
        }
    }

    #[test]
    fn the_vocabulary_is_all_grammar_edits_and_nothing_else() {
        let mut d = GrammarDomain::new(grammar_ops(), Admission::new(2), 1, 3, |_| 0.0);
        let state = PolicyState::default();
        let codes = d.legal(&state);
        assert!(!codes.is_empty());
        for c in codes {
            match d.instantiate(&c, &state) {
                Some(Mutation::GrammarEdit { .. }) => {}
                other => panic!("level 2 must only play grammar edits, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_level_two_search_scores_a_grammar_by_the_search_below_it() {
        // The inner closure stands in for a level-1 search. Here it rewards a
        // deeper rollout budget, so the level-2 search should discover and keep
        // an edit that raises the depth.
        let (score, mutations) = search_grammar(Admission::new(2), 7, 3, &budget(), |s| {
            s.grammar().depth() as f64
        });
        assert!(score >= Grammar::default().depth() as f64, "score {score}");
        assert!(
            !mutations.is_empty(),
            "the search should have found grammar edits worth keeping"
        );
        for m in &mutations {
            assert!(matches!(m, Mutation::GrammarEdit { .. }));
        }
    }

    #[test]
    fn the_search_finds_the_grammar_the_inner_search_prefers() {
        // A sharper version: the inner search only likes a reweight delta of
        // exactly 2.0. The level-2 search has to find that specific edit.
        let (score, mutations) = search_grammar(Admission::new(2), 11, 3, &budget(), |s| {
            if (s.grammar().reweight_delta() - 2.0).abs() < 1e-9 { 1.0 } else { 0.0 }
        });
        assert_eq!(score, 1.0, "the preferred grammar should be found");
        let applied: Vec<GrammarOp> = mutations
            .iter()
            .filter_map(|m| match m {
                Mutation::GrammarEdit { spec } => serde_json::from_value(spec.clone()).ok(),
                _ => None,
            })
            .collect();
        assert!(
            applied
                .iter()
                .any(|op| matches!(op, GrammarOp::SetReweightDelta { value } if (*value - 2.0).abs() < 1e-9)),
            "expected the winning trace to set the delta the inner search wanted: {applied:?}"
        );
    }

    #[test]
    fn with_level_two_disabled_the_search_can_change_nothing() {
        // The frozen gate is upstream of the whole level. Every edit is refused,
        // so the search runs and finds nothing — it does not error, and it does
        // not sneak a change through.
        let (_score, mutations) = search_grammar(Admission::new(1), 3, 3, &budget(), |s| {
            s.grammar().depth() as f64
        });
        assert!(
            mutations.is_empty(),
            "no grammar edit may take effect with level 2 off: {mutations:?}"
        );
    }

    #[test]
    fn no_grammar_op_in_the_vocabulary_can_promote_the_search() {
        // Belt and braces on the level barrier: the shipped vocabulary contains
        // no op that names a level-3 move, so a level-2 search cannot reach one
        // even by exhausting its whole move set.
        for op in grammar_ops() {
            if let GrammarOp::EnableKind { kind } = &op {
                assert_ne!(kind, "code_patch", "the vocabulary must not offer level 3");
                assert_ne!(kind, "grammar_edit");
            }
        }
    }
}
