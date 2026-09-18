"""Tests for the paired-eval statistics. No GPU, no model.

These decide whether a number gets called a result, so they are the part that
must not be quietly wrong. The failure that matters is a confident interval
around noise - which is exactly what resampling the wrong unit produces.

    python training/test_eval_paired.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paired import (  # noqa: E402
    bootstrap_ci,
    informative,
    paired_difference,
)

K = 8

# ------------------------------------------------------- mean difference ---

assert paired_difference([4, 4], [4, 4], K) == 0.0, "no change is zero"
assert paired_difference([0, 0], [8, 8], K) == 1.0, "0 -> 8 of 8 is +100%"
assert paired_difference([8, 8], [0, 0], K) == -1.0
assert paired_difference([], [], K) == 0.0, "empty must not divide by zero"

# Per-PROBLEM rates, not pooled samples. One problem going 0 -> 8 while another
# goes 8 -> 0 is a mean change of zero, and pooling would say the same - but the
# per-problem form is what the bootstrap below needs.
assert paired_difference([0, 8], [8, 0], K) == 0.0
print(f"mean difference: {paired_difference([2, 3], [4, 6], K):+.1%} for +2 and +3 of 8")

# ------------------------------------------------------------- bootstrap ---

# THE test. The unit of independence is the PROBLEM: eight rollouts of one
# problem are eight looks at the same question, not eight looks at the model.
# Resampling rollouts instead of problems narrows the interval by roughly
# sqrt(8) and turns noise into a finding.
same = [4] * 20
lo, hi = bootstrap_ci(same, same, K)
assert lo == 0.0 == hi, f"identical arms must give a zero-width interval: {lo},{hi}"
print("identical arms -> interval is exactly zero")

# A real, consistent effect: every problem improves by one of eight.
lo, hi = bootstrap_ci([4] * 20, [5] * 20, K)
assert lo > 0, f"a consistent +1/8 on every problem must exclude zero: {lo}"
print(f"consistent +1/8 on 20 problems -> {lo:+.1%} to {hi:+.1%}, excludes zero")

# Noise: half the problems up by one, half down by one. The mean is zero and the
# interval MUST span it, or the test manufactures findings.
rng = random.Random(7)
noisy_b = [4] * 30
noisy_t = [4 + (1 if i % 2 else -1) for i in range(30)]
lo, hi = bootstrap_ci(noisy_b, noisy_t, K)
assert lo < 0 < hi, f"symmetric noise must span zero: {lo} to {hi}"
print(f"symmetric noise -> {lo:+.1%} to {hi:+.1%}, spans zero")

# A small effect on few problems must NOT come out significant. This is the
# shape of the run that produced 2 fixed and 0 broken.
lo, hi = bootstrap_ci([0, 0, 8, 8, 8], [1, 1, 8, 8, 8], K)
assert lo <= 0, f"two tiny gains on five problems is not a result: {lo} to {hi}"
print(f"tiny effect, 5 problems -> {lo:+.1%} to {hi:+.1%}, does not exclude zero")

# Same effect size, many more problems: now it should resolve. This is the whole
# argument for a bigger, harder eval.
lo, hi = bootstrap_ci([4] * 60, [5] * 60, K)
assert lo > 0, "the same effect on 60 problems should resolve"
print("the difference between the two is N, not the effect")

# One problem cannot have an interval resampled out of it. It must say so with
# nan rather than return a confident-looking zero-width range.
lo, hi = bootstrap_ci([4], [5], K)
assert lo != lo and hi != hi, f"n=1 must be nan, not {lo} to {hi}"
lo, hi = bootstrap_ci([], [], K)
assert lo != lo and hi != hi, "n=0 must be nan too"
print("one problem yields nan rather than a fake interval")

# ----------------------------------------------------------- informative ---

# An item the base always solves can only get worse; one it never solves can
# only get better. Neither can show a small improvement, and an eval made of
# them measures nothing - which is what 38 of 40 items were last time.
assert informative([0, 8, 4, 7, 1], K) == 3, "0 and 8 of 8 are decided"
assert informative([0] * 10, K) == 0
assert informative([8] * 10, K) == 0
print("informative(): only items the base solves SOMETIMES count")

print("\npaired-eval statistics verified - no GPU, no model")
