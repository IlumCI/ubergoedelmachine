"""Tests for the GRPO reward-health diagnostic.

It fires ten batches into a paid GPU session, so it gets tested on a laptop
rather than discovered there. Needs no GPU and no training dependencies:

    python training/test_reward_health.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_reasoning import reward_health, HEALTH_WINDOW

DONE = "reasoning...\nAnswer: 42\nConfidence: 0.9"
CUT = "reasoning that never finishes, and never reaches a closing line"

# 1. Healthy: groups disagree.
m = reward_health([DONE] * 8, [[True, False]] * 4, 6144)
assert "something to learn from" in m, m
assert not m.startswith("***"), m
print("healthy  ->", m)

# 2. Dead by truncation: unanimous AND almost nothing finished. Must point at
#    the budget, not the difficulty - the fixes are opposite.
m = reward_health([CUT] * 8, [[False] * 8] * 10, 2048)
assert m.startswith("***") and "CUT OFF" in m and "6144" not in m, m
assert "--max-completion above 2048" in m, m
print("truncated->", m)

# 3. Dead by difficulty: unanimous but the rollouts did finish.
m = reward_health([DONE] * 8, [[True] * 8] * 10, 6144)
assert m.startswith("***") and "DIFFICULTY" in m, m
assert "CUT OFF" not in m, m
print("too easy ->", m)

# 4. Unanimously wrong but finishing is still a difficulty problem, not a budget one.
m = reward_health([DONE] * 8, [[False] * 8] * 10, 6144)
assert "DIFFICULTY" in m and "CUT OFF" not in m, m
print("too hard ->", m)

# 5. The boundary: exactly half finished counts as "finishing".
m = reward_health([DONE] * 4 + [CUT] * 4, [[False] * 8] * 10, 6144)
assert "DIFFICULTY" in m, m

# 6. Degenerate input must not divide by zero.
m = reward_health([], [[False]] * 10, 6144)
assert m.startswith("***"), m

# 7. The case pooling gets WRONG, and the reason spread is counted per group:
#    five all-right groups and five all-wrong ones pool to a perfectly mixed
#    set, yet every single group is unanimous, so the advantage is zero and
#    there is no gradient at all.
m = reward_health([DONE] * 80, [[True] * 8] * 5 + [[False] * 8] * 5, 6144)
assert m.startswith("***"), f"pooled spread masked ten flat groups: {m}"
print("flat-but-pooled ->", m)

# 8. One mixed group among nine flat ones still counts as alive.
m = reward_health([DONE] * 80, [[True, False] + [True] * 6] + [[True] * 8] * 9, 6144)
assert not m.startswith("***") and "1/10" in m, m
print("one mixed ->", m)

assert HEALTH_WINDOW == 10
print("\nreward_health OK - all 8 cases")
