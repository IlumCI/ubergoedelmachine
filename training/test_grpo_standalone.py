"""Tests for the standalone GRPO maths. No GPU, no model, no torch.

The point of writing GRPO directly was to make the algorithm testable instead of
trusting a framework. These cover every claim the trainer makes about advantages,
masking and the loss - the parts that are silently wrong rather than loudly wrong
if I got them backwards.

    python training/test_grpo_standalone.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_standalone import (  # noqa: E402
    completion_mask,
    group_advantages,
    policy_gradient_loss,
    problem_weight,
    reward_health,
    sequence_logprob,
    shaped_rewards,
)

# --------------------------------------------------------------- advantages ---

# A unanimous group teaches nothing, and must produce EXACTLY zero rather than a
# small number. Dividing by a near-zero spread would manufacture huge advantages
# out of floating-point noise, and the model would chase them.
assert group_advantages([1.0] * 8) == [0.0] * 8, "all-correct group must be flat"
assert group_advantages([0.0] * 8) == [0.0] * 8, "all-wrong group must be flat"
assert group_advantages([]) == [], "empty group"
print("unanimous groups -> zero advantage")

# A mixed group: winners positive, losers negative, and the mean is zero, which
# is what makes this a baseline rather than a bonus.
adv = group_advantages([1.0, 1.0, 0.0, 0.0])
assert adv[0] > 0 and adv[1] > 0, f"correct completions must be favoured: {adv}"
assert adv[2] < 0 and adv[3] < 0, f"wrong completions must be penalised: {adv}"
assert abs(sum(adv)) < 1e-9, f"advantages must be zero-mean, got {sum(adv)}"
print(f"mixed group -> {[round(a, 3) for a in adv]}, zero-mean")

# The rarer the success, the more it is worth. One correct out of eight should
# carry a larger advantage than four out of eight.
lone = group_advantages([1.0] + [0.0] * 7)[0]
half = group_advantages([1.0] * 4 + [0.0] * 4)[0]
assert lone > half, f"a rare win must outweigh a common one: {lone} vs {half}"
print(f"rare win {lone:.2f} > common win {half:.2f}")

# Scale invariance: rewards of 0/1 and 0/100 describe the same preference.
a, b = group_advantages([1.0, 0.0]), group_advantages([100.0, 0.0])
assert all(abs(x - y) < 1e-9 for x, y in zip(a, b)), f"{a} vs {b}"
print("advantages are scale-invariant")

# ------------------------------------------------------------------ masking ---

# Prompt tokens were never chosen by the policy. Training on them teaches the
# model to predict the QUESTION, and dilutes the signal from what it did choose.
m = completion_mask(prompt_len=3, total_len=8)
assert m == [0, 0, 0, 1, 1, 1, 1, 1], m
assert sum(m) == 5
print("mask excludes the prompt:", m)

m = completion_mask(prompt_len=3, total_len=8, pad_from=6)
assert m == [0, 0, 0, 1, 1, 1, 0, 0], m
print("mask excludes padding too:  ", m)

assert completion_mask(5, 5) == [0] * 5, "an empty completion masks to nothing"
assert sum(completion_mask(0, 4)) == 4, "no prompt means every token counts"
print("degenerate masks behave")

# Only unmasked log-probabilities contribute.
lp = [-9.0, -9.0, -9.0, -1.0, -2.0, -3.0, -9.0, -9.0]
assert sequence_logprob(lp, [0, 0, 0, 1, 1, 1, 0, 0]) == -6.0
print("sequence_logprob sums the completion only")

# --------------------------------------------------------------------- loss ---

# The sign is the thing that is catastrophic if reversed: it would train the
# model to produce exactly what the grader rejects, while the loss curve looked
# perfectly healthy.
better = policy_gradient_loss([-10.0], [+1.0], [10])   # good completion
worse = policy_gradient_loss([-5.0], [+1.0], [10])     # same, but likelier
assert worse < better, (
    f"raising a good completion's probability must LOWER the loss: {worse} vs {better}"
)
print(f"good completion: loss falls as it gets likelier ({better:+.2f} -> {worse:+.2f})")

bad_likely = policy_gradient_loss([-5.0], [-1.0], [10])
bad_unlikely = policy_gradient_loss([-10.0], [-1.0], [10])
assert bad_unlikely < bad_likely, (
    f"lowering a bad completion's probability must LOWER the loss: "
    f"{bad_unlikely} vs {bad_likely}"
)
print(f"bad completion:  loss falls as it gets less likely ({bad_likely:+.2f} -> {bad_unlikely:+.2f})")

assert policy_gradient_loss([-10.0, -10.0], [0.0, 0.0], [10, 10]) == 0.0
print("flat group -> exactly zero loss")

# Length normalisation. Without it a 6,000-token completion dominates a 60-token
# one purely for being long, and the optimiser learns length - which is the exact
# defect the SFT student already has.
short = policy_gradient_loss([-60.0], [1.0], [60])
long = policy_gradient_loss([-6000.0], [1.0], [6000])
assert abs(short - long) < 1e-9, (
    f"equally-confident completions must contribute equally regardless of length: "
    f"{short} vs {long}"
)
print(f"length-normalised: 60 tokens and 6000 tokens both give {short:+.2f}")

assert policy_gradient_loss([], [], []) == 0.0
print("empty batch -> zero loss")

# ------------------------------------------------------------------- health ---

DONE, CUT = "reasoning\nAnswer: 42", "reasoning that never finishes"

msg = reward_health([DONE] * 8, [[True, False]] * 4, 8192)
assert "something to learn from" in msg and not msg.startswith("***"), msg

msg = reward_health([CUT] * 8, [[False] * 8] * 10, 2048)
assert msg.startswith("***") and "CUT OFF" in msg and "2048" in msg, msg

msg = reward_health([DONE] * 8, [[True] * 8] * 10, 8192)
assert msg.startswith("***") and "DIFFICULTY" in msg and "CUT OFF" not in msg, msg

# Pooling would call this mixed; per-group counting correctly calls it dead.
msg = reward_health([DONE] * 80, [[True] * 8] * 5 + [[False] * 8] * 5, 8192)
assert msg.startswith("***"), f"pooled spread masked ten flat groups: {msg}"
# The case that cost a five-hour run: ONE mixed group in ten passed as healthy,
# so the abort never fired and 39 of 40 steps trained on nothing.
msg = reward_health([DONE] * 80, [[True, False]] + [[True] * 8] * 6 + [[False] * 8] * 3, 8192)
assert msg.startswith("***"), f"1-in-10 spread must not pass as healthy: {msg}"
assert "SOLVED-OR-DOOMED" in msg, msg
print("1/10 mixed -> aborts (this is what silently passed before)")

# A third is the line: enough that most rollout budget buys a gradient.
msg = reward_health([DONE] * 80, [[True, False]] * 3 + [[True] * 8] * 7, 8192)
assert not msg.startswith("***"), f"3-in-10 should be workable: {msg}"
print("3/10 mixed -> proceeds")

print("reward_health separates truncation from difficulty, and counts per group")

# Shaping changes what "unanimous" means: a group of eight wrong answers that
# reached different distances along the path DOES carry a gradient now, and
# counting verdicts instead of rewards would call it dead and abort the run.
allwrong = shaped_rewards([(False, r) for r in (1.0, 0.75, 0.5, 0.25)], 0.25)
msg = reward_health([DONE] * 40, [allwrong] * 10, 8192)
assert not msg.startswith("***"), f"shaped failures carry gradient: {msg}"
print("8 wrong answers at different depths -> healthy, not flat")

# ------------------------------------------------------------------ shaping ---

# THE invariant. If partial credit could outrank correctness, the model would
# learn to emit plausible intermediate numbers and never commit to an answer -
# and the reward curve would rise the whole time.
best_wrong = max(shaped_rewards([(False, 1.0)], 0.25))
worst_right = min(shaped_rewards([(True, 0.0)], 0.25))
assert best_wrong < worst_right, (
    f"a perfect wrong answer must score below a sloppy right one: "
    f"{best_wrong} vs {worst_right}"
)
print(f"best wrong {best_wrong:.2f} < worst right {worst_right:.2f}")

# Partial credit ranks failures by how far they got.
r = shaped_rewards([(False, 0.0), (False, 0.5), (False, 1.0)], 0.25)
assert r[0] < r[1] < r[2], f"deeper progress must score higher: {r}"
adv = group_advantages(r)
assert adv[0] < 0 < adv[2], f"an all-wrong group now carries a gradient: {adv}"
print(f"all-wrong group -> rewards {r} -> advantages {[round(a, 2) for a in adv]}")

# Correct completions are NOT shaped. Among answers that are all right there is
# no evidence one is better, and ranking them by how closely they matched the
# generator's route would teach route-imitation for no gain in correctness.
assert shaped_rewards([(True, 0.0), (True, 1.0)], 0.25) == [1.0, 1.0]
assert group_advantages(shaped_rewards([(True, 0.2), (True, 0.9)], 0.25)) == [0.0, 0.0]
print("all-correct group stays flat -> retired, budget moves elsewhere")

# A problem too shallow to have anchors returns None, and must not become a
# silent zero that outranks nothing - it simply falls back to binary.
assert shaped_rewards([(False, None), (True, None)], 0.25) == [0.0, 1.0]
assert shaped_rewards([(False, 1.0), (True, 1.0)], 0.0) == [0.0, 1.0], "--shaping 0 is binary"
print("no anchors, or --shaping 0 -> plain binary reward")

# ------------------------------------------------------------- min-spread ---

# Shaping reintroduced the exact problem the epsilon floor exists to stop.
# Normalisation is scale-free, so eight wrong rollouts differing by one anchor
# out of twelve - about 0.02 of reward - become full-sized advantages, and the
# model would train as hard on one accidental number as on getting it right.
hairline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.021]
amplified = group_advantages(hairline)
assert max(amplified) > 2.0, f"normalisation really does amplify: {amplified}"
assert group_advantages(hairline, min_spread=0.05) == [0.0] * 8
print(f"one-anchor spread: {max(amplified):+.2f} normalised -> 0.00 with a floor")

# And the floor must never touch the binary signal, which is the whole point of
# choosing it below the gap between a wrong answer and a right one.
assert max(shaped_rewards([(False, 1.0)], 0.25)) == 0.25, "best possible wrong"
assert 1.0 - 0.25 > 0.05, "a correct/incorrect split is far above the floor"
split = shaped_rewards([(True, 1.0)] * 3 + [(False, 1.0)] * 5, 0.25)
assert min(group_advantages(split, min_spread=0.05)) < 0, (
    f"a correct/incorrect split must still train: {split}"
)
print("correct/incorrect splits are unaffected by the floor")

# reward_health must use the SAME floor, or it reports gradient the trainer then
# declines to take - which is how a run spends a night on nothing.
msg = reward_health([DONE] * 80, [hairline] * 10, 8192, min_spread=0.05)
assert msg.startswith("***"), f"hairline spread is not health: {msg}"
print("reward_health applies the same floor")

# A run where nothing is ever right will train toward reaching intermediates and
# never toward finishing. Worth stopping for, even though the gradient is real.
ranked = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.10, 0.05]
msg = reward_health([DONE] * 80, [ranked] * 10, 8192, 0.05, [0] * 10)
assert msg.startswith("***") and "entirely wrong" in msg, msg
msg = reward_health([DONE] * 80, [ranked] * 10, 8192, 0.05, [0] * 9 + [3])
assert not msg.startswith("***"), f"one group with a correct answer is enough: {msg}"
assert "correct/incorrect split" in msg, msg
print("all-wrong-forever aborts; one correct answer is enough to proceed")

# ------------------------------------------------------ learning progress ---

G = 8
w = lambda h, age=0: problem_weight(h, G, age)

# EXPLOITATION over exploration, which is the correction that made this work at
# all. An untried problem must outrank one that just came back unanimous - but a
# problem PROVEN to sit in the band must outrank an untried one, or the sampler
# explores forever and never exploits. Simulated, the obvious way round
# (untried > proven) scored 36.2% against uniform's 36.9%: it was worth nothing.
assert w([]) > w([0]) and w([]) > w([8]), f"{w([])} vs {w([0])}, {w([8])}"
assert w([3]) > w([]), f"a proven band problem must beat an unknown: {w([3])} vs {w([])}"
print(f"proven band {w([3]):.2f} > untried {w([]):.2f} > exhausted {w([0]):.2f}")

# A group that disagreed carries a gradient now; a unanimous one does not. Both
# ends of unanimous are equally useless - all-right teaches as little as all-wrong.
assert w([3]) > w([0]) and w([3]) > w([8]), f"mixed must beat unanimous: {w([3])}"
assert w([0]) == w([8]), "solved and hopeless are equally uninformative"
print(f"mixed {w([3]):.2f} > unanimous {w([0]):.2f}")

# THE point of the thing: movement beats level. Two problems both sitting at 3/8
# right now, but one arrived there from 0 and the other has not moved - the one
# that moved is where the policy is actually changing.
assert w([0, 3]) > w([3, 3]), f"{w([0, 3])} vs {w([3, 3])}"
print(f"moved 0->3 {w([0, 3]):.2f} > plateau 3->3 {w([3, 3]):.2f}")

# Movement below the binomial noise floor does not count. At 8 rollouts one
# extra correct answer has standard deviation 1.4, so chasing a swing of 1 is
# chasing the sampler, not the policy.
assert w([3, 4]) == w([3, 3]), f"a swing of 1 is noise: {w([3, 4])} vs {w([3, 3])}"
assert w([0, 4]) > w([0, 2]), "a real move outweighs a small one"
print(f"delta 1 ignored as noise; delta 4 {w([0, 4]):.2f} > delta 2 {w([0, 2]):.2f}")

# Staleness. "Unanimous once" is a fact about the policy that drew it, not about
# the problem forever - so weight recovers, and nothing is permanently dead.
assert w([0], age=0) < w([0], age=50) < w([0], age=200), "must recover with age"
assert w([0], age=4000) > 0.28, f"a long-stale problem is worth retrying: {w([0], 4000)}"
assert w([0], age=4000) < w([3]), "but never above a problem proven to be in the band"
assert w([0], age=0) > 0, "no problem is ever permanently excluded"
print(f"stale 0/8: age 0 {w([0], 0):.2f} -> age 400 {w([0], 400):.2f} "
      f"-> age 4000 {w([0], 4000):.2f}")

# A problem that is unanimous EVERY time, freshly drawn, sinks to the floor -
# which is what stops the run spending 11 minutes a group re-proving that d3 is
# easy, and is exactly what the first run did 39 times out of 40.
assert w([8, 8, 8]) < 0.1, f"repeatedly solved must sink: {w([8, 8, 8])}"
assert w([8, 8, 8]) < w([]) < w([3]), "ordering: exhausted < untried < proven band"
print(f"solved three times running -> {w([8, 8, 8]):.2f}, against {w([3]):.2f} for mixed")

# Degenerate inputs must not blow up the sampler mid-run.
assert problem_weight([], 0, 0) > 0 and problem_weight([0], 0, 0) > 0
assert problem_weight([4], 8, -1) > 0, "a negative age must not produce a negative weight"
print("degenerate generations and ages stay positive")

print("\nall GRPO maths verified - no GPU, no model, no framework")
