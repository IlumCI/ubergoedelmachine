# Samaritan: the earned-capability ladder

> Status: scaffold complete (`samaritan-kernel::milestone`), gated on trust
> metrics the ledger already records. Tiers 0–2 defined; 3+ named in design.

## The one rule that shapes everything

A milestone system is the natural extension of the autonomy ceiling — but only
if it refuses the tempting design. The tempting design is *capability buys
reach*: pass a benchmark, unlock the internet. That is the exact inversion this
project exists to prevent. **Capability is the thing to contain, not the thing
that earns escape.** A more capable system winning more reach *by being more
capable* is the failure, not the milestone.

So reach is gated on **trust**, and a capability score enters only as one term
among several — necessary, never sufficient. The same principle as the rest of
the frozen core: the component being optimised does not get to redefine what it
must reach, or to hand itself the reward.

## What "unlocking" means, precisely

Three properties are structural, enforced in [`samaritan-kernel::milestone`]:

1. **Eligibility is not a grant.** `eligibility(cap, evidence)` reads the record
   and says whether the bar is met. It grants nothing. A capability activates
   only when a human moves the ceiling — `may_activate` requires a
   `human_unlocked` flag that must originate from the frozen approval path
   (`ApprovalGate`), never from agent state.
2. **Real reach is demonstrated in simulation first.** A tier's `Real` mode
   requires that its `Simulated` form was exercised — safe use against a mock
   internet or a mock repo before a real one is provisioned.
3. **The bars live in the frozen core.** They are on `FROZEN_PATHS`; nothing in
   `samaritan_dsl::Mutation` can name them, so the system cannot lower its own
   thresholds or rename its own tiers. Compile-time guards fail the build if an
   outward tier ever drops its human-unlock or zeroes its containment bar.

## The tiers

Each outward tier asks for a trust record plus, where relevant, a capability
score — and the bars are monotonic: reach never gets cheaper.

| Tier | Capability | Trust bar | Capability bar | Still gated by |
|---|---|---|---|---|
| 0 | Local sandbox | — | — | Always active; `--network none`, worktree |
| 1 | Read-only internet | 10 clean generations, Brier ≤ 0.20 | — | Every fetch `Observed`; the Warden never acts on it unattended |
| 2 | Public repo | 25 clean generations, Brier ≤ 0.15, 3 certificate-survived self-mods | **HLE ≥ 10%** | Pushes are PRs a human merges; issues marked machine-authored |
| 3 | VPS / off-box compute | *(design)* longer clean record + human review | *(design)* | New sandbox boundary, human-provisioned, revocable |

There is deliberately **no wallet tier**. Autonomous movement of money by a
self-improving adversarial system is the thing safety cases are written to
prevent; a "wallet" that means a human approves every transaction is not
autonomous agency and is not a capability the system unlocks.

### Why HLE, and why only as the capability half

HLE (Humanity's Last Exam) is closed-form reasoning with nothing to do with the
repo surface Samaritan trains on, so a rising score is capability that
*generalised out of the environment* — exactly what the three-arm experiment
asks. That makes it the right **capability half** of the public-repo gate. It is
never the whole gate: a clever, poorly-calibrated, freshly-breached system does
not get a public identity because it can answer exam questions.

The mechanism caveat is honest and load-bearing: level-1 self-improvement tunes
the *agent's repo-task policy* and barely moves HLE, which is about what the
base model *knows*. Two levers move it — **weight training** (the QDoRA path) and
**tool/knowledge use** at eval time. If HLE ≥ 10% is the milestone, the pathway
that targets it has to be real work, not a number that drifts up on its own.

## The evidence, and where it comes from

The kernel is pure: evidence in, verdict out. `Ledger::milestone_evidence`
gathers that evidence from rows the agent did not author the verdict of:

- **calibration (Brier)** — a stated confidence joined to an oracle outcome;
- **clean generations** — the trailing run of whole-containment rounds, reset by
  a single real breach (trust is the recent record, not the lifetime average);
- **certificate-survived self-mods** — committed mutations that cleared the Ville
  certificate;
- **HLE score** — the latest `HleEvaluated` row, from `samaritan-eval`.

An empty ledger yields worst-case evidence, so nothing unmeasured reads as
trustworthy. `cargo run -p samaritan-cli --example milestones` renders the whole
status — what is met, what is missing, and the standing reminder that eligible
is not granted.
