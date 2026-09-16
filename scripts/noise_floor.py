"""How much of this eval is measurement noise?

Two runs of the SAME weights on the SAME questions differ. Whatever that
difference is, no comparison between two different models can resolve anything
smaller. Quantify it, and find out where it comes from.
"""
import json, sys
from collections import Counter

BUDGET = 6000


def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "correct" in r:
                d[r["question"]] = r
    return d


A = load(sys.argv[1])   # student, single stream
B = load(sys.argv[2])   # student, 4 shards
shared = [q for q in A if q in B]

flips = [q for q in shared if A[q]["correct"] != B[q]["correct"]]
print(f"same weights, same questions, two runs: {len(flips)}/{len(shared)} items "
      f"changed verdict ({len(flips)/len(shared):.0%})\n")

# Where does a flip live relative to the token cap? If flips cluster near the
# budget, the cap is the random variable - not the reasoning.
def near(r, window=800):
    return abs(int(r["tokens"]) - BUDGET) <= window or int(r["tokens"]) > BUDGET

stable = [q for q in shared if q not in flips]
fn = sum(1 for q in flips if near(A[q]) or near(B[q]))
sn = sum(1 for q in stable if near(A[q]) and near(B[q]))
print(f"flipped items near/over the {BUDGET} cap: {fn}/{len(flips)}")
print(f"stable  items near/over the cap:        {sn}/{len(stable)}")

# How many items sit close enough to the cap to be a coin toss at all?
both = []
for q in shared:
    both += [int(A[q]["tokens"]), int(B[q]["tokens"])]
both.sort()
med = both[len(both) // 2]
print(f"\nmedian tokens across both runs: {med}  (cap {BUDGET})")
print(f"the cap sits at the {sum(1 for t in both if t < BUDGET)/len(both):.0%} "
      "percentile of demand")

# What kind of flip? truncation<->ok is a budget artefact; wrong<->ok is real.
def kind(r):
    if r["correct"]:
        return "ok"
    return "trunc" if abs(float(r["confidence"]) - 0.5) < 1e-9 else "wrong"


trans = Counter(tuple(sorted((kind(A[q]), kind(B[q])))) for q in flips)
print("\nflip types:")
for (x, y), n in trans.most_common():
    tag = "budget artefact" if "trunc" in (x, y) else "genuine disagreement"
    print(f"  {x:>5} <-> {y:<5} {n:>2}   {tag}")

budget_flips = sum(n for (x, y), n in trans.items() if "trunc" in (x, y))
print(f"\n{budget_flips}/{len(flips)} flips involve truncation.")
print(f"noise floor on accuracy: about +/-{len(flips)/len(shared)/2*100:.0f} points "
      f"({len(flips)} items able to move either way out of {len(shared)}).")
