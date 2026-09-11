//! The bridge from mined candidates to the nested search.
//!
//! [`mine`](crate::mine) produces candidate mutations from a run's outcomes;
//! [`samaritan_search`] knows how to search over mutations but not where they
//! come from. This is the join: a [`SelfModDomain`] whose legal moves *are*
//! the mined candidates, so the level-1 NRPA search explores combinations of
//! evidence-backed lessons rather than a fixed grammar of guesses.
//!
//! The key restraint: the search may only combine and order what mining
//! proposed. It cannot invent a lesson, because a lesson with no evidence is
//! exactly the self-authored kind this crate exists to avoid. The candidate
//! set is the search's whole vocabulary, and every word in it came from the
//! ledger.
//!
//! Scoring a candidate policy means running episodes with it, which needs the
//! agent and a live model. That is injected as a closure so the *search
//! wiring* is testable here without inference — the part that has to be right
//! and the part that is expensive to run are deliberately separable.

use std::collections::HashMap;

use samaritan_dsl::{Mutation, MutationCode, MutationPolicy};
use samaritan_kernel::Admission;
use samaritan_search::{Domain, PolicyState};

use crate::Candidate;

/// Scores a policy state by whatever means the caller supplies — in a live
/// run, by playing episodes against held-out tasks and returning mean utility.
pub type Evaluator<'a> = dyn FnMut(&PolicyState) -> f64 + 'a;

/// A [`Domain`] over mined self-modifications.
pub struct SelfModDomain<'a> {
    /// The mined vocabulary, indexed by a stable code. The search may pick
    /// from these and nothing else.
    candidates: HashMap<MutationCode, Candidate>,
    /// Insertion order, so a rollout is deterministic given the RNG.
    order: Vec<MutationCode>,
    admission: Admission,
    evaluate: Box<Evaluator<'a>>,
    rng: u64,
    depth: usize,
}

impl<'a> SelfModDomain<'a> {
    /// Build a domain from a mined candidate set.
    ///
    /// `depth` is how many mutations a rollout may stack — a lesson set is
    /// usually small, so a handful is plenty and more just re-treads.
    pub fn new(
        candidates: Vec<Candidate>,
        admission: Admission,
        seed: u64,
        depth: usize,
        evaluate: impl FnMut(&PolicyState) -> f64 + 'a,
    ) -> Self {
        let mut map = HashMap::new();
        let mut order = Vec::new();
        for (i, c) in candidates.into_iter().enumerate() {
            // The code carries the index so distinct candidates that happen to
            // render similar text stay distinct moves.
            let code = MutationCode(format!("cand:{i}:{}", short(&c.mutation)));
            order.push(code.clone());
            map.insert(code, c);
        }
        Self {
            candidates: map,
            order,
            admission,
            evaluate: Box::new(evaluate),
            rng: seed.max(1),
            depth,
        }
    }

    /// Which candidate a code names, for a caller reading a finished trace.
    pub fn candidate(&self, code: &MutationCode) -> Option<&Candidate> {
        self.candidates.get(code)
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

fn short(m: &Mutation) -> String {
    match m {
        Mutation::LessonAdd { .. } => "lesson_add".into(),
        Mutation::LessonRemove { .. } => "lesson_remove".into(),
        Mutation::LessonReweight { .. } => "lesson_reweight".into(),
        Mutation::ThresholdSet { knob, .. } => format!("threshold_{knob:?}"),
        Mutation::GrammarEdit { .. } => "grammar_edit".into(),
        Mutation::CodePatch { .. } => "code_patch".into(),
    }
}

impl Domain for SelfModDomain<'_> {
    fn legal(&self, state: &PolicyState) -> Vec<MutationCode> {
        // A candidate is legal until it has already been applied in this
        // rollout, which shows up as its effect already being present. Cheap
        // proxy: a lesson whose text is already a lesson in the state is spent.
        self.order
            .iter()
            .filter(|code| {
                let Some(cand) = self.candidates.get(*code) else {
                    return false;
                };
                match &cand.mutation {
                    Mutation::LessonAdd { text } => {
                        !state.lessons().iter().any(|l| &l.text == text)
                    }
                    _ => true,
                }
            })
            .cloned()
            .collect()
    }

    fn instantiate(&mut self, code: &MutationCode, _state: &PolicyState) -> Option<Mutation> {
        self.candidates.get(code).map(|c| c.mutation.clone())
    }

    fn apply(&mut self, state: &mut PolicyState, m: &Mutation) -> bool {
        state.apply(m, &self.admission).is_ok()
    }

    fn evaluate(&mut self, state: &PolicyState) -> f64 {
        (self.evaluate)(state)
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn random(&mut self) -> f64 {
        self.next_unit()
    }
}

/// Run a level-1 search over a mined candidate set and return the winning
/// mutation sequence.
///
/// A thin, honest wrapper: mining supplies the vocabulary, the search supplies
/// the ordering, the caller's evaluator supplies the ground truth, and none
/// of the three is trusted to do another's job. Whether the winner may be
/// *committed* is still `samaritan-cert`'s call, on held-out tasks — this only
/// finds the candidate.
pub fn search_mined<'a>(
    candidates: Vec<Candidate>,
    admission: Admission,
    seed: u64,
    depth: usize,
    budget: &samaritan_search::Budget,
    evaluate: impl FnMut(&PolicyState) -> f64 + 'a,
) -> (f64, Vec<Mutation>) {
    let mut domain = SelfModDomain::new(candidates, admission, seed, depth, evaluate);
    let base = PolicyState::default();
    let (score, trace) = samaritan_search::nrsi(
        1,
        &MutationPolicy::new(),
        &mut domain,
        &base,
        budget,
    );
    (score, trace.mutations)
}
