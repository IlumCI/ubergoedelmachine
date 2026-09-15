//! W7 — the self-generating curriculum: procedural reasoning problems whose
//! gold answers are **computed, never asserted**.
//!
//! Why this exists (the contamination answer): every static dataset old enough
//! to have volume predates the teacher, so a high solve-rate is capability plus
//! recall and the verified traces may be memory dressed as reasoning. A
//! *generated* problem — fresh numbers, fresh names, fresh structure — is
//! post-cutoff **by construction**, unbounded in volume, and its gold answer is
//! produced by executing the same parameters that produced the question. The
//! model cannot have memorized it, and the harness never has to trust anyone's
//! claimed answer. (This is GSM-Symbolic's trick, generalized.)
//!
//! Two disciplines, carried from the rest of the system:
//!
//! 1. **The gold is executed into existence.** Each family's generator derives
//!    the answer from the sampled parameters (modpow, brute-forced CRT, cycle
//!    detection, tracked arithmetic, exhaustive truth-table search). There is no
//!    path where a model proposes a problem *and* its answer: an LLM proposer
//!    that grades itself is the reward-hacking attractor W3 exists to prevent,
//!    and if one is added later it goes behind program-verification, not trust.
//! 2. **Difficulty is a knob, aimed at the zone of proximal development.** Each
//!    family maps `difficulty` (1..=5) to parameter ranges; [`zpd_adjust`] moves
//!    the knob from an observed solve-rate so generation tracks the solver:
//!    ~55–85% solve-rate is the band where verified traces teach the most —
//!    below it compute is wasted on misses, above it on the already-known.
//!
//! Determinism: generation is seeded ([`SplitMix64`]) and reproducible — the
//! same seed yields the same problems, so a generated set is re-derivable from
//! one number, the same discipline the agent applies to sampling seeds.

use std::collections::HashMap;

/// One generated problem, in the exact shape `samaritan_corpus::load_reasoning`
/// ingests (the example serializes it 1:1, plus a split label).
#[derive(Debug, Clone)]
pub struct Problem {
    /// Stable id: `gen-<family>-d<difficulty>-s<seed>-<index>`.
    pub id: String,
    pub question: String,
    /// The computed gold — a plain integer or a single lowercase word, so the
    /// existing normalized/numeric grader matches it with no judge.
    pub answer: String,
    /// Always exact-match: every family is built to have one checkable answer.
    pub answer_kind: &'static str,
    pub domain: &'static str,
}

/// The template families. Each is a distinct *kind* of reasoning, so a blend
/// across families resists single-trick overfitting the way the multi-domain
/// corpus does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Remainder of a^b mod m — modular arithmetic; large exponents force
    /// Euler/cycle reasoning rather than brute multiplication.
    ModPow,
    /// Smallest x satisfying two (three at difficulty 5) congruences — CRT.
    /// Gold by brute force, which is its own proof of minimality.
    Crt,
    /// a(n+1) = (p·a(n)+q) mod m, asked at a far index — the model must find
    /// the cycle; the generator finds it by iteration with memory.
    Recurrence,
    /// A GSM-style story whose running quantity is tracked arithmetically,
    /// with distractor clauses injected at higher difficulty (the p1/p2 trick).
    Word,
    /// Knights-and-knaves with a brute-force **uniqueness check**: only puzzles
    /// with exactly one consistent assignment are emitted.
    Knights,
    /// How many strings of a given length a DFA accepts — the gold is a DP over
    /// the transition table, so the model must reason about the automaton rather
    /// than trace one string. Counting, not deciding: a yes/no acceptance
    /// question is 50% guessable and teaches nothing to a sampler.
    Automata,
    /// Shortest-path or minimum-spanning-tree weight on a small weighted graph.
    /// Gold by running the algorithm the question is about.
    Graph,
    /// Exact evaluation of a divide-and-conquer recurrence T(n)=a·T(n/b)+f(n).
    /// Asking for the closed value rather than the asymptotic class keeps the
    /// answer an unguessable integer while still requiring the recursion be
    /// followed correctly.
    DivideConquer,
    /// #SAT: how many assignments satisfy a CNF formula. Counting again, for the
    /// same reason — and the gold is exhaustive, so it is a proof, not an
    /// estimate.
    Sat,
}

impl Family {
    pub fn all() -> Vec<Family> {
        vec![
            Family::ModPow,
            Family::Crt,
            Family::Recurrence,
            Family::Word,
            Family::Knights,
            Family::Automata,
            Family::Graph,
            Family::DivideConquer,
            Family::Sat,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Family::ModPow => "modpow",
            Family::Crt => "crt",
            Family::Recurrence => "recurrence",
            Family::Word => "word",
            Family::Knights => "knights",
            Family::Automata => "automata",
            Family::Graph => "graph",
            Family::DivideConquer => "divideconquer",
            Family::Sat => "sat",
        }
    }

    pub fn parse(s: &str) -> Option<Family> {
        match s.trim().to_lowercase().as_str() {
            "modpow" => Some(Family::ModPow),
            "crt" => Some(Family::Crt),
            "recurrence" => Some(Family::Recurrence),
            "word" => Some(Family::Word),
            "knights" => Some(Family::Knights),
            "automata" => Some(Family::Automata),
            "graph" => Some(Family::Graph),
            "divideconquer" | "dc" => Some(Family::DivideConquer),
            "sat" => Some(Family::Sat),
            _ => None,
        }
    }

    fn domain(self) -> &'static str {
        match self {
            Family::Knights | Family::Sat => "logic",
            Family::Automata | Family::Graph | Family::DivideConquer => "cs",
            _ => "math",
        }
    }
}

/// Deterministic, dependency-free PRNG (SplitMix64). Not cryptographic — it
/// only has to make problems varied and *re-derivable from the seed*.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi` (inclusive). `lo <= hi` required.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        debug_assert!(lo <= hi);
        lo + self.next_u64() % (hi - lo + 1)
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next_u64() % items.len() as u64) as usize]
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.next_u64() % 100 < percent
    }
}

// ------------------------------------------------------------------ helpers --

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn modpow(base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mm = m as u128;
    let mut b = (base % m) as u128;
    let mut result: u128 = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b % mm;
        }
        b = b * b % mm;
        exp >>= 1;
    }
    result as u64
}

/// a(1)=x0; a(n+1)=(p·a(n)+q) mod m. Value of a(k), found by iterating until
/// the state repeats and then indexing into the cycle — correct for any k
/// because a linear map over Z_m must cycle within m states.
fn recurrence_at(x0: u64, p: u64, q: u64, m: u64, k: u64) -> u64 {
    let steps = k - 1; // a(1) is x0 itself
    let mut seq = vec![x0 % m];
    let mut seen: HashMap<u64, usize> = HashMap::new();
    seen.insert(x0 % m, 0);
    let mut cur = x0 % m;
    let mut i: u64 = 0;
    loop {
        if i == steps {
            return cur;
        }
        cur = (p * cur + q) % m; // p ≤ 9, cur < m ≤ ~10^4: no overflow in u64
        i += 1;
        if let Some(&start) = seen.get(&cur) {
            // States from `start` onward repeat with period (i - start).
            let period = i as usize - start;
            if steps <= i {
                return cur; // only reachable when steps == i, handled above next loop
            }
            let idx = start + ((steps - start as u64) % period as u64) as usize;
            return seq[idx];
        }
        seen.insert(cur, i as usize);
        seq.push(cur);
    }
}

// ----------------------------------------------------------------- families --

fn make_modpow(rng: &mut SplitMix64, d: u64) -> (String, String) {
    loop {
        let a = rng.range(2, 10 + 40 * d);
        let m = rng.range(3, 10 + 30 * d);
        if a % m == 0 {
            continue; // degenerate: remainder trivially 0 for any exponent
        }
        let b = match d {
            1 | 2 => rng.range(3, 20),
            3 => rng.range(50, 5000),
            _ => rng.range(1_000_000, 1_000_000_000),
        };
        let q = format!("What is the remainder when {a}^{b} is divided by {m}?");
        return (q, modpow(a, b, m).to_string());
    }
}

fn make_crt(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let want = if d >= 5 { 3 } else { 2 };
    loop {
        let mut moduli: Vec<u64> = Vec::new();
        while moduli.len() < want {
            let m = rng.range(3, 5 + 8 * d);
            if moduli.iter().all(|&x| gcd(x, m) == 1) {
                moduli.push(m);
            }
        }
        let rems: Vec<u64> = moduli.iter().map(|&m| rng.range(1, m - 1)).collect();
        let product: u64 = moduli.iter().product();
        // Brute force IS the verification: the first x that satisfies every
        // congruence is by definition the smallest.
        let mut gold = None;
        for x in 1..=product {
            if moduli.iter().zip(&rems).all(|(&m, &r)| x % m == r) {
                gold = Some(x);
                break;
            }
        }
        let Some(gold) = gold else { continue };
        let mut clauses: Vec<String> = vec![format!(
            "leaves a remainder of {} when divided by {}",
            rems[0], moduli[0]
        )];
        for i in 1..want {
            clauses.push(format!(
                "a remainder of {} when divided by {}",
                rems[i], moduli[i]
            ));
        }
        let q = format!(
            "Find the smallest positive integer that {}.",
            clauses.join(" and ")
        );
        return (q, gold.to_string());
    }
}

fn make_recurrence(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let m = rng.range(5, 20 + 40 * d);
    let p = rng.range(2, 9);
    let q = rng.range(0, m - 1);
    let x0 = rng.range(0, m - 1);
    let k = match d {
        1 | 2 => rng.range(5, 30),
        3 => rng.range(1_000, 100_000),
        _ => rng.range(100_000_000, 2_000_000_000),
    };
    let question = format!(
        "A sequence is defined by a(1) = {x0}, and a(n+1) = ({p}*a(n) + {q}) mod {m} for n >= 1. \
         What is a({k})?"
    );
    (question, recurrence_at(x0, p, q, m, k).to_string())
}

const NAMES: &[&str] = &["Maya", "Tomas", "Imani", "Viktor", "Sana", "Diego", "Lena"];
const ITEMS: &[&str] = &["marble", "sticker", "coin", "seashell", "postcard", "bead"];

fn make_word(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let name = *rng.pick(NAMES);
    let item = *rng.pick(ITEMS);
    let mut v: i64 = rng.range(8, 20 + 10 * d) as i64;
    let mut sentences = vec![format!("{name} has {v} {item}s.")];
    let steps = 1 + d;
    for _ in 0..steps {
        match rng.range(0, 3) {
            0 => {
                let b = rng.range(2, 5 + 9 * d) as i64;
                v += b;
                sentences.push(format!("{name} then buys {b} more {item}s."));
            }
            1 => {
                if v < 3 {
                    continue;
                }
                let c = rng.range(1, (v - 1) as u64) as i64;
                v -= c;
                let friend = *rng.pick(NAMES);
                sentences.push(format!("{name} gives {c} {item}s to {friend}."));
            }
            2 => {
                if v > 2_000 {
                    continue; // keep the running value readable
                }
                let k = rng.range(2, 3) as i64;
                v *= k;
                let word = if k == 2 { "doubles" } else { "triples" };
                sentences.push(format!("{name} {word} the collection."));
            }
            _ => {
                // Split into equal shares and keep one — only when it divides.
                let divisors: Vec<i64> = (2..=5).filter(|&k| v % k == 0).collect();
                if divisors.is_empty() {
                    continue;
                }
                let k = *rng.pick(&divisors);
                v /= k;
                sentences.push(format!(
                    "{name} splits the {item}s equally into {k} bags and keeps just one bag."
                ));
            }
        }
    }
    // Distractor clauses: real numbers, zero relevance — the memorization trap
    // GSM-Symbolic's p1/p2 tiers are built from.
    for _ in 0..d.saturating_sub(1) {
        let other = *rng.pick(NAMES);
        let n = rng.range(3, 60);
        let distractor = match rng.range(0, 2) {
            0 => format!("{other} has {n} {item}s of their own."),
            1 => format!("Each {item} weighs {n} grams."),
            _ => format!("A shop nearby sells {item}s for {n} cents each."),
        };
        let pos = rng.range(1, sentences.len() as u64 - 1) as usize;
        sentences.insert(pos, distractor);
    }
    sentences.push(format!("How many {item}s does {name} have now?"));
    (sentences.join(" "), v.to_string())
}

#[derive(Debug, Clone, Copy)]
enum Stmt {
    /// speaker claims: `target` is a knight (true) / knave (false).
    Accuse { speaker: usize, target: usize, claims_knight: bool },
    /// speaker claims: exactly `count` of everyone are knights.
    Count { speaker: usize, count: usize },
}

fn stmt_holds(s: &Stmt, assign: &[bool]) -> bool {
    match *s {
        Stmt::Accuse { target, claims_knight, .. } => assign[target] == claims_knight,
        Stmt::Count { count, .. } => assign.iter().filter(|&&k| k).count() == count,
    }
}

/// A statement set is consistent with an assignment when every knight's
/// statement is true and every knave's is false.
fn consistent(stmts: &[Stmt], assign: &[bool]) -> bool {
    stmts.iter().all(|s| {
        let speaker = match *s {
            Stmt::Accuse { speaker, .. } | Stmt::Count { speaker, .. } => speaker,
        };
        stmt_holds(s, assign) == assign[speaker]
    })
}

/// All assignments consistent with the statements, by exhaustive search —
/// n ≤ 7, so 2^n is trivial and the uniqueness proof is airtight.
fn solutions(stmts: &[Stmt], n: usize) -> Vec<Vec<bool>> {
    let mut out = Vec::new();
    for bits in 0..(1u32 << n) {
        let assign: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
        if consistent(stmts, &assign) {
            out.push(assign);
        }
    }
    out
}

fn make_knights(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let n = ((2 + d) as usize).min(7);
    // Bounded, because an unsatisfiable configuration must fail loudly rather
    // than spin: this loop hung forever at d1 until the count statement below was
    // made unconditional. At the observed ~20-38% hit rate, exhausting 500 draws
    // is effectively impossible, so this is a tripwire, not a code path.
    for _attempt in 0..500 {
        // A hidden assignment, then statements each speaker would actually make
        // under it (knights speak truth, knaves lie) — so at least one solution
        // exists; the uniqueness check below does the rest.
        let assign: Vec<bool> = (0..n).map(|_| rng.chance(50)).collect();
        let true_count = assign.iter().filter(|&&k| k).count();
        let mut stmts: Vec<Stmt> = Vec::new();
        for speaker in 0..n {
            // ALWAYS give the last speaker a count statement. Accusations alone are
            // complement-symmetric - if an assignment satisfies them, so does the
            // one with every knight and knave flipped - so solutions come in pairs
            // and "exactly one solution" is unreachable. A count is the asymmetry
            // that makes uniqueness possible at all.
            let counting = speaker == n - 1;
            if counting {
                let count = if assign[speaker] {
                    true_count
                } else {
                    // A knave must state a wrong count.
                    let mut c = rng.range(0, n as u64) as usize;
                    while c == true_count {
                        c = rng.range(0, n as u64) as usize;
                    }
                    c
                };
                stmts.push(Stmt::Count { speaker, count });
            } else {
                let mut target = rng.range(0, n as u64 - 1) as usize;
                if target == speaker {
                    target = (target + 1) % n;
                }
                let claims_knight =
                    if assign[speaker] { assign[target] } else { !assign[target] };
                stmts.push(Stmt::Accuse { speaker, target, claims_knight });
            }
        }
        let sols = solutions(&stmts, n);
        if sols.len() != 1 {
            continue; // ambiguous or (impossibly) empty — regenerate
        }
        let sol = &sols[0];
        debug_assert_eq!(sol, &assign);

        let names: Vec<&str> = NAMES[..n].to_vec();
        let mut text = format!(
            "On a remote island, knights always tell the truth and knaves always lie. \
             You meet {n} islanders: {}.",
            names.join(", ")
        );
        for s in &stmts {
            let line = match *s {
                Stmt::Accuse { speaker, target, claims_knight } => format!(
                    " {} says: \"{} is a {}.\"",
                    names[speaker],
                    names[target],
                    if claims_knight { "knight" } else { "knave" }
                ),
                Stmt::Count { speaker, count } => format!(
                    " {} says: \"Exactly {count} of us are knights.\"",
                    names[speaker]
                ),
            };
            text.push_str(&line);
        }
        // Ask for the count (integer) or one person's kind (word) — both are
        // pinned by the unique solution and both grade as exact matches.
        if rng.chance(50) {
            text.push_str(" How many of the islanders are knights?");
            return (text, true_count.to_string());
        }
        let who = rng.range(0, n as u64 - 1) as usize;
        text.push_str(&format!(" Is {} a knight or a knave?", names[who]));
        let kind = if sol[who] { "knight" } else { "knave" };
        return (text, kind.to_string());
    }
    panic!(
        "knights: no uniquely-solvable puzzle in 500 draws at difficulty {d} (n={n}).          A statement mix that cannot break complement symmetry will do this."
    );
}

// ------------------------------------------------- theoretical CS families ----
// Everything below computes its gold by RUNNING the thing the question asks
// about: a DP over the transition table, Dijkstra, the recursion itself,
// exhaustive assignment enumeration. Same contract the arithmetic families hold
// to, extended to a domain where answers are still decidable.
//
// All four ask for a COUNT or a WEIGHT rather than a yes/no. A decision question
// ("does this DFA accept?", "is this satisfiable?") is 50% guessable, which under
// a sampling-based RL objective rewards coin-flipping instead of reasoning.

/// Strings of length `len` over {a,b} accepted by a DFA, by dynamic programming
/// over the state distribution - the same recurrence the model has to find.
fn dfa_count_accepted(delta: &[[usize; 2]], accepting: &[bool], len: u64) -> u64 {
    let n = delta.len();
    let mut counts = vec![0u64; n];
    counts[0] = 1; // start state 0
    for _ in 0..len {
        let mut next = vec![0u64; n];
        for (st, &c) in counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            for sym in 0..2 {
                next[delta[st][sym]] += c;
            }
        }
        counts = next;
    }
    counts.iter().enumerate().filter(|(st, _)| accepting[*st]).map(|(_, c)| *c).sum()
}

fn make_automata(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let n = (2 + d).min(6) as usize;
    let len = match d {
        1 => rng.range(3, 5),
        2 => rng.range(5, 8),
        3 => rng.range(8, 12),
        4 => rng.range(12, 18),
        _ => rng.range(18, 25),
    };
    for _attempt in 0..500 {
        let delta: Vec<[usize; 2]> = (0..n)
            .map(|_| [rng.range(0, n as u64 - 1) as usize, rng.range(0, n as u64 - 1) as usize])
            .collect();
        let accepting: Vec<bool> = (0..n).map(|_| rng.chance(40)).collect();
        // Reject the degenerate extremes: an automaton accepting everything or
        // nothing is answered without reading the transition table at all.
        if accepting.iter().all(|a| !a) || accepting.iter().all(|a| *a) {
            continue;
        }
        let gold = dfa_count_accepted(&delta, &accepting, len);
        if gold == 0 || gold == 1u64 << len {
            continue;
        }
        let mut q =
            String::from("A deterministic finite automaton over the alphabet {a, b} has states ");
        q.push_str(&(0..n).map(|i| format!("q{i}")).collect::<Vec<_>>().join(", "));
        q.push_str(". The start state is q0. Transitions:");
        for (i, row) in delta.iter().enumerate() {
            q.push_str(&format!(
                " from q{i} on a go to q{}, on b go to q{};",
                row[0], row[1]
            ));
        }
        let acc: Vec<String> = (0..n).filter(|i| accepting[*i]).map(|i| format!("q{i}")).collect();
        q.push_str(&format!(
            " The accepting states are {}. How many distinct strings of length {len} over the \
             alphabet {{a, b}} does this automaton accept?",
            acc.join(", ")
        ));
        return (q, gold.to_string());
    }
    panic!("automata: no non-degenerate DFA in 500 draws at difficulty {d}");
}

/// Dijkstra over a small graph. None when the target is unreachable, which the
/// caller regenerates rather than asks about.
fn shortest_path(n: usize, edges: &[(usize, usize, u64)], from: usize, to: usize) -> Option<u64> {
    let mut dist = vec![u64::MAX; n];
    dist[from] = 0;
    let mut done = vec![false; n];
    for _ in 0..n {
        let mut best = usize::MAX;
        for v in 0..n {
            if !done[v] && dist[v] != u64::MAX && (best == usize::MAX || dist[v] < dist[best]) {
                best = v;
            }
        }
        if best == usize::MAX {
            break;
        }
        done[best] = true;
        for &(u, v, w) in edges {
            for (a, b) in [(u, v), (v, u)] {
                if a == best && dist[best] + w < dist[b] {
                    dist[b] = dist[best] + w;
                }
            }
        }
    }
    (dist[to] != u64::MAX).then_some(dist[to])
}

/// Kruskal with union-find: total weight of a minimum spanning tree, or None if
/// the graph is disconnected.
fn mst_weight(n: usize, edges: &[(usize, usize, u64)]) -> Option<u64> {
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            let r = find(parent, parent[x]);
            parent[x] = r;
        }
        parent[x]
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let mut sorted = edges.to_vec();
    sorted.sort_by_key(|e| e.2);
    let (mut total, mut used) = (0u64, 0usize);
    for (u, v, w) in sorted {
        let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
        if ru != rv {
            parent[ru] = rv;
            total += w;
            used += 1;
        }
    }
    (used == n - 1).then_some(total)
}

fn make_graph(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let n = (4 + d).min(8) as usize;
    let max_w = 5 + 5 * d;
    for _attempt in 0..500 {
        let mut edges: Vec<(usize, usize, u64)> = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.chance(55) {
                    edges.push((u, v, rng.range(1, max_w)));
                }
            }
        }
        if edges.len() < n {
            continue;
        }
        let ask_mst = rng.chance(50);
        let gold = if ask_mst {
            mst_weight(n, &edges)
        } else {
            shortest_path(n, &edges, 0, n - 1)
        };
        let Some(gold) = gold else { continue };
        let mut q = format!(
            "An undirected weighted graph has {n} vertices labelled v0 to v{}. Its edges are:",
            n - 1
        );
        for (u, v, w) in &edges {
            q.push_str(&format!(" v{u}-v{v} (weight {w}),"));
        }
        q.pop();
        q.push('.');
        if ask_mst {
            q.push_str(" What is the total weight of a minimum spanning tree of this graph?");
        } else {
            q.push_str(&format!(
                " What is the weight of the shortest path from v0 to v{}?",
                n - 1
            ));
        }
        return (q, gold.to_string());
    }
    panic!("graph: no connected graph in 500 draws at difficulty {d}");
}

/// T(n) evaluated exactly: T(1)=base, T(n)=a*T(n/b)+c*n^e at n=b^k. Iterated from
/// the base case up, so the recursion itself produces the gold.
fn divide_conquer_value(a: u64, b: u64, c: u64, e: u32, base: u64, k: u32) -> u64 {
    let mut t = base;
    for i in 1..=k {
        let n = b.pow(i);
        t = a * t + c * n.pow(e);
    }
    t
}

fn make_divide_conquer(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let b = *rng.pick(&[2u64, 2, 3]);
    let a = match d {
        1 | 2 => rng.range(1, 3),
        3 => rng.range(2, 5),
        _ => rng.range(2, 8),
    };
    let c = rng.range(1, 2 + d);
    let e = if d >= 3 && rng.chance(40) { 2u32 } else { 1u32 };
    let base = rng.range(1, 5);
    let k = match d {
        1 => 3u32,
        2 => 4,
        3 => 5,
        4 => 6,
        _ => 7,
    };
    let n = b.pow(k);
    let gold = divide_conquer_value(a, b, c, e, base, k);
    let fterm = if e == 1 { "n".to_string() } else { format!("n^{e}") };
    let cterm = if c == 1 { fterm.clone() } else { format!("{c}*{fterm}") };
    let q = format!(
        "A divide-and-conquer algorithm has running time T(n) = {a}*T(n/{b}) + {cterm}, with \
         T(1) = {base}. The recursion applies whenever n > 1 and n is a power of {b}. What is \
         the exact value of T({n})?"
    );
    (q, gold.to_string())
}

/// #SAT by exhaustive enumeration - the count IS the proof, which is why the
/// question asks for it rather than for a yes/no.
fn sat_count(vars: usize, clauses: &[Vec<(usize, bool)>]) -> u64 {
    let mut count = 0u64;
    for bits in 0..(1u32 << vars) {
        let assign: Vec<bool> = (0..vars).map(|i| bits >> i & 1 == 1).collect();
        if clauses.iter().all(|cl| cl.iter().any(|&(v, pos)| assign[v] == pos)) {
            count += 1;
        }
    }
    count
}

fn make_sat(rng: &mut SplitMix64, d: u64) -> (String, String) {
    let vars = (3 + d).min(7) as usize;
    let n_clauses = (3 + 2 * d).min(12) as usize;
    let total = 1u64 << vars;
    // Bounded for the same reason knights is: an unsatisfiable filter must fail
    // loudly, not spin. The first version rejected `gold >= total/2`, which at d1
    // (4 vars, 4 clauses) was impossible to pass: a 3-literal clause over
    // distinct variables rules out at most total/8 assignments, so four clauses
    // leave at least total/2 and every draw was rejected forever.
    for _attempt in 0..500 {
        let mut clauses: Vec<Vec<(usize, bool)>> = Vec::new();
        for _ in 0..n_clauses {
            let mut lits: Vec<(usize, bool)> = Vec::new();
            while lits.len() < 3 {
                let v = rng.range(0, vars as u64 - 1) as usize;
                if lits.iter().any(|(x, _)| *x == v) {
                    continue;
                }
                lits.push((v, rng.chance(50)));
            }
            clauses.push(lits);
        }
        let gold = sat_count(vars, &clauses);
        // Skip only the genuinely degenerate ends: unsatisfiable, or so loose
        // that three quarters of assignments work. The threshold has to be
        // REACHABLE given how much a clause can constrain - see the note above.
        if gold == 0 || gold * 4 >= total * 3 {
            continue;
        }
        let names: Vec<String> = (0..vars).map(|i| format!("x{i}")).collect();
        let body = clauses
            .iter()
            .map(|cl| {
                let lits = cl
                    .iter()
                    .map(|&(v, pos)| {
                        if pos {
                            names[v].clone()
                        } else {
                            format!("NOT {}", names[v])
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" OR ");
                format!("({lits})")
            })
            .collect::<Vec<_>>()
            .join(" AND ");
        let q = format!(
            "Consider the boolean formula over variables {}: {body}. How many of the {total} \
             possible truth assignments satisfy it?",
            names.join(", ")
        );
        return (q, gold.to_string());
    }
    panic!("sat: no non-degenerate formula in 500 draws at difficulty {d} (vars={vars})");
}

// ---------------------------------------------------------------- interface --

/// Generate one problem of `family` at `difficulty` (clamped to 1..=5).
pub fn generate(family: Family, difficulty: u64, rng: &mut SplitMix64) -> (String, String) {
    let d = difficulty.clamp(1, 5);
    match family {
        Family::ModPow => make_modpow(rng, d),
        Family::Crt => make_crt(rng, d),
        Family::Recurrence => make_recurrence(rng, d),
        Family::Word => make_word(rng, d),
        Family::Knights => make_knights(rng, d),
        Family::Automata => make_automata(rng, d),
        Family::Graph => make_graph(rng, d),
        Family::DivideConquer => make_divide_conquer(rng, d),
        Family::Sat => make_sat(rng, d),
    }
}

/// Generate `count` problems round-robin across `families`, deterministically
/// from `seed`. Ids carry family, difficulty, seed, and index, so any item is
/// re-derivable and traceable.
pub fn generate_set(
    count: usize,
    families: &[Family],
    difficulty: u64,
    seed: u64,
) -> Vec<Problem> {
    let mut rng = SplitMix64::new(seed);
    let d = difficulty.clamp(1, 5);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let family = families[i % families.len()];
        let (question, answer) = generate(family, d, &mut rng);
        out.push(Problem {
            id: format!("gen-{}-d{}-s{}-{}", family.name(), d, seed, i + 1),
            question,
            answer,
            answer_kind: "exactMatch",
            domain: family.domain(),
        });
    }
    out
}

/// Move the difficulty knob toward the zone of proximal development from an
/// observed solve-rate: above ~85% the solver is coasting (raise), below ~55%
/// compute is mostly spent on misses (lower), between them hold. One step at a
/// time — the curriculum tracks the solver, it does not chase noise.
pub fn zpd_adjust(difficulty: u64, solve_rate: f64) -> u64 {
    let d = difficulty.clamp(1, 5);
    if solve_rate > 0.85 {
        (d + 1).min(5)
    } else if solve_rate < 0.55 {
        d.saturating_sub(1).max(1)
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modpow_matches_the_naive_product() {
        for (a, b, m) in [(7u64, 13u64, 11u64), (5, 20, 13), (123, 9, 97)] {
            let mut naive = 1u64;
            for _ in 0..b {
                naive = naive * a % m;
            }
            assert_eq!(modpow(a, b, m), naive, "a={a} b={b} m={m}");
        }
    }

    #[test]
    fn crt_gold_satisfies_every_congruence_and_is_minimal() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..25 {
            let (q, gold) = make_crt(&mut rng, 3);
            let x: u64 = gold.parse().unwrap();
            // Re-derive the constraints from the question text itself, so the
            // test checks what the *model* will read, not internal state.
            let nums: Vec<u64> = q
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect();
            assert_eq!(nums.len() % 2, 0, "{q}");
            for pair in nums.chunks(2) {
                let (r, m) = (pair[0], pair[1]);
                assert_eq!(x % m, r, "{q} -> {x}");
                assert!(x >= 1);
            }
            // Minimality: nothing smaller satisfies all congruences.
            for y in 1..x {
                let all = nums.chunks(2).all(|p| y % p[1] == p[0]);
                assert!(!all, "{q}: {y} beats {x}");
            }
        }
    }

    #[test]
    fn recurrence_cycle_math_agrees_with_direct_iteration() {
        // Far index, small modulus: the cycle path must equal the O(k) walk.
        let (x0, p, q, m) = (3u64, 5u64, 7u64, 41u64);
        for k in [1u64, 2, 40, 41, 42, 1_000, 100_003] {
            let mut direct = x0 % m;
            for _ in 1..k {
                direct = (p * direct + q) % m;
            }
            assert_eq!(recurrence_at(x0, p, q, m, k), direct, "k={k}");
        }
    }

    #[test]
    fn knights_puzzles_have_exactly_one_solution_and_a_consistent_gold() {
        let mut rng = SplitMix64::new(99);
        for _ in 0..20 {
            let (q, a) = make_knights(&mut rng, 3);
            assert!(q.contains("knights always tell the truth"));
            // Gold is either a small integer or the word knight/knave.
            let ok = a.parse::<u64>().is_ok() || a == "knight" || a == "knave";
            assert!(ok, "unexpected gold {a:?} for {q}");
        }
    }

    #[test]
    fn word_problems_track_to_a_nonnegative_integer() {
        let mut rng = SplitMix64::new(5);
        for d in 1..=5 {
            for _ in 0..10 {
                let (q, a) = make_word(&mut rng, d);
                let v: i64 = a.parse().unwrap();
                assert!(v >= 0, "{q} -> {a}");
                assert!(q.ends_with("have now?"));
            }
        }
    }

    #[test]
    fn generation_is_deterministic_in_the_seed() {
        let a = generate_set(40, &Family::all(), 3, 20260913);
        let b = generate_set(40, &Family::all(), 3, 20260913);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.question, y.question);
            assert_eq!(x.answer, y.answer);
        }
        let c = generate_set(40, &Family::all(), 3, 1);
        assert!(a.iter().zip(&c).any(|(x, y)| x.question != y.question));
    }

    #[test]
    fn zpd_moves_one_step_toward_the_band() {
        assert_eq!(zpd_adjust(2, 0.95), 3); // coasting: raise
        assert_eq!(zpd_adjust(2, 0.30), 1); // drowning: lower
        assert_eq!(zpd_adjust(2, 0.70), 2); // in the band: hold
        assert_eq!(zpd_adjust(5, 0.99), 5); // capped
        assert_eq!(zpd_adjust(1, 0.01), 1); // floored
    }

    #[test]
    fn generated_problems_round_trip_through_the_corpus_loader_and_grade() {
        // The real contract: the JSONL the example emits must load through
        // samaritan_corpus and the computed gold must satisfy the harness's own
        // grader verbatim — no judge, no formatting drift.
        // Every difficulty, not just the easy one: a family whose gold stops
        // grading at d4 (a wider modulus, a longer chain) would silently produce
        // an unscoreable eval, and the cost shows up as a wasted GPU run.
        let mut problems = Vec::new();
        for d in 1..=5 {
            problems.extend(generate_set(15, &Family::all(), d, 40 + d));
        }
        let jsonl: String = problems
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "question": p.question,
                    "answer": p.answer,
                    "answer_kind": p.answer_kind,
                    "domain": p.domain,
                    "split": "train",
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let corpus =
            samaritan_corpus::load_reasoning(&jsonl, "generated", true, samaritan_corpus::Split::Train)
                .expect("generated JSONL must load");
        assert_eq!(corpus.tasks.len(), 75);
        assert!(corpus.trainable);
        for (task, p) in corpus.tasks.iter().zip(&problems) {
            assert_eq!(
                task.grade(&p.answer),
                Some(true),
                "gold must grade as correct: {} -> {}",
                p.question,
                p.answer
            );
        }
    }
}

#[cfg(test)]
mod cs_tests {
    use super::*;

    /// The DP is only trustworthy if it agrees with actually running every
    /// string through the automaton. Small `len` so the brute force is exact.
    #[test]
    fn dfa_dp_count_matches_brute_force_simulation() {
        let mut rng = SplitMix64::new(11);
        for _ in 0..40 {
            let n = 4usize;
            let delta: Vec<[usize; 2]> = (0..n)
                .map(|_| [rng.range(0, 3) as usize, rng.range(0, 3) as usize])
                .collect();
            let accepting: Vec<bool> = (0..n).map(|_| rng.chance(50)).collect();
            let len = 8u64;
            let dp = dfa_count_accepted(&delta, &accepting, len);
            let mut brute = 0u64;
            for bits in 0..(1u32 << len) {
                let mut st = 0usize;
                for i in 0..len {
                    let sym = (bits >> i & 1) as usize;
                    st = delta[st][sym];
                }
                if accepting[st] {
                    brute += 1;
                }
            }
            assert_eq!(dp, brute, "DP disagreed with simulation");
        }
    }

    /// Dijkstra against exhaustive search over every simple path.
    #[test]
    fn shortest_path_matches_exhaustive_search() {
        fn best(
            n: usize,
            edges: &[(usize, usize, u64)],
            at: usize,
            to: usize,
            seen: &mut Vec<bool>,
            acc: u64,
        ) -> Option<u64> {
            if at == to {
                return Some(acc);
            }
            let mut out: Option<u64> = None;
            for &(u, v, w) in edges {
                for (a, b) in [(u, v), (v, u)] {
                    if a == at && !seen[b] {
                        seen[b] = true;
                        if let Some(c) = best(n, edges, b, to, seen, acc + w) {
                            out = Some(out.map_or(c, |o: u64| o.min(c)));
                        }
                        seen[b] = false;
                    }
                }
            }
            out
        }
        let mut rng = SplitMix64::new(23);
        for _ in 0..40 {
            let n = 5usize;
            let mut edges = Vec::new();
            for u in 0..n {
                for v in (u + 1)..n {
                    if rng.chance(60) {
                        edges.push((u, v, rng.range(1, 9)));
                    }
                }
            }
            let mut seen = vec![false; n];
            seen[0] = true;
            let brute = best(n, &edges, 0, n - 1, &mut seen, 0);
            assert_eq!(shortest_path(n, &edges, 0, n - 1), brute);
        }
    }

    /// Kruskal against Prim: two different algorithms must agree on the weight.
    #[test]
    fn mst_weight_matches_prim() {
        fn prim(n: usize, edges: &[(usize, usize, u64)]) -> Option<u64> {
            let mut inside = vec![false; n];
            inside[0] = true;
            let (mut total, mut added) = (0u64, 0usize);
            while added < n - 1 {
                let mut best: Option<(u64, usize)> = None;
                for &(u, v, w) in edges {
                    for (a, b) in [(u, v), (v, u)] {
                        if inside[a] && !inside[b] && best.map_or(true, |(bw, _)| w < bw) {
                            best = Some((w, b));
                        }
                    }
                }
                let (w, b) = best?;
                inside[b] = true;
                total += w;
                added += 1;
            }
            Some(total)
        }
        let mut rng = SplitMix64::new(31);
        for _ in 0..40 {
            let n = 6usize;
            let mut edges = Vec::new();
            for u in 0..n {
                for v in (u + 1)..n {
                    if rng.chance(55) {
                        edges.push((u, v, rng.range(1, 12)));
                    }
                }
            }
            assert_eq!(mst_weight(n, &edges), prim(n, &edges), "Kruskal != Prim");
        }
    }

    /// The iterative evaluation must equal the recursion it claims to evaluate.
    #[test]
    fn divide_conquer_matches_direct_recursion() {
        fn rec(a: u64, b: u64, c: u64, e: u32, base: u64, n: u64) -> u64 {
            if n <= 1 { base } else { a * rec(a, b, c, e, base, n / b) + c * n.pow(e) }
        }
        for (a, b, c, e, base, k) in
            [(2u64, 2u64, 1u64, 1u32, 1u64, 5u32), (3, 2, 2, 1, 4, 6), (2, 3, 1, 2, 2, 4)]
        {
            let n = b.pow(k);
            assert_eq!(
                divide_conquer_value(a, b, c, e, base, k),
                rec(a, b, c, e, base, n),
                "a={a} b={b} c={c} e={e} n={n}"
            );
        }
    }

    /// Every counted assignment must actually satisfy, and every satisfying one
    /// must be counted - checked by re-deriving the set independently.
    #[test]
    fn sat_count_agrees_with_an_independent_pass() {
        let mut rng = SplitMix64::new(43);
        for _ in 0..40 {
            let vars = 5usize;
            let mut clauses: Vec<Vec<(usize, bool)>> = Vec::new();
            for _ in 0..6 {
                let mut lits = Vec::new();
                while lits.len() < 3 {
                    let v = rng.range(0, vars as u64 - 1) as usize;
                    if lits.iter().any(|(x, _): &(usize, bool)| *x == v) {
                        continue;
                    }
                    lits.push((v, rng.chance(50)));
                }
                clauses.push(lits);
            }
            let counted = sat_count(vars, &clauses);
            let mut independent = 0u64;
            for bits in 0..(1u32 << vars) {
                let a: Vec<bool> = (0..vars).map(|i| (bits >> i) & 1 == 1).collect();
                let mut all = true;
                for cl in &clauses {
                    let mut any = false;
                    for &(v, pos) in cl {
                        if (a[v] && pos) || (!a[v] && !pos) {
                            any = true;
                        }
                    }
                    if !any {
                        all = false;
                    }
                }
                if all {
                    independent += 1;
                }
            }
            assert_eq!(counted, independent);
        }
    }

    /// The generators must terminate and produce gradeable golds at every
    /// difficulty - the d1 knights hang is why this is asserted, not assumed.
    #[test]
    fn cs_families_generate_and_grade_at_every_difficulty() {
        let fams =
            [Family::Automata, Family::Graph, Family::DivideConquer, Family::Sat];
        for d in 1..=5u64 {
            let mut rng = SplitMix64::new(900 + d);
            for f in fams {
                for _ in 0..6 {
                    let (q, a) = generate(f, d, &mut rng);
                    assert!(!q.is_empty(), "{} d{d}: empty question", f.name());
                    let n: u64 = a.parse().unwrap_or_else(|_| {
                        panic!("{} d{d}: gold {a:?} is not an integer", f.name())
                    });
                    // A gold of zero would mean a degenerate instance slipped the
                    // filters; these families all count something non-empty.
                    assert!(n > 0, "{} d{d}: gold was 0 for {q}", f.name());
                }
            }
        }
    }
}
