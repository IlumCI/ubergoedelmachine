"""Paired analysis of two reason_eval progress files.

Splits failures into TRUNCATIONS and GENUINE WRONG ANSWERS before comparing,
because they are different phenomena that a single accuracy number merges: one
says the model ran out of room, the other says it reasoned badly, and only the
second is a capability claim.
"""
import json, sys, math
from collections import Counter

# NOTE ON TOKEN COUNTS: reason_eval sums tokens across retry attempts, so a
# single item can report more than the per-call budget. That makes a cap
# impossible to infer reliably from the data, so no cap is inferred - the
# distribution is reported instead, which is what actually distinguishes a model
# that finishes from one that runs away.


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "correct" not in r:
            continue
        rows[r["question"]] = r
    return rows


def kind(r):
    """A 0.5 confidence is reason_eval's marker for 'no Confidence line parsed',
    which in practice means the reply was cut off before it could close."""
    if r["correct"]:
        return "ok"
    return "truncated" if abs(float(r["confidence"]) - 0.5) < 1e-9 else "wrong"


a_path, b_path = sys.argv[1], sys.argv[2]
a_name, b_name = sys.argv[3], sys.argv[4]
A, B = load(a_path), load(b_path)

shared = [q for q in A if q in B]
print(f"{a_name}: {len(A)} items | {b_name}: {len(B)} items | matched: {len(shared)}\n")

for name, D in ((a_name, A), (b_name, B)):
    ks = Counter(kind(r) for r in D.values())
    n = len(D)
    tok = sorted(int(r["tokens"]) for r in D.values())
    print(f"{name}:")
    print(f"  accuracy   {ks['ok']}/{n} = {ks['ok']/n:.0%}")
    print(f"  truncated  {ks['truncated']}  (conf 0.5, ran out of room)")
    print(f"  wrong      {ks['wrong']}  (finished and got it wrong)")
    p90 = tok[int(len(tok) * 0.9)]
    print(f"  median tok {tok[len(tok)//2]}   p90 {p90}   max {tok[-1]}")
    runaway = sum(1 for t in tok if t > 3 * tok[len(tok) // 2])
    print(f"  runaway (>3x median): {runaway}  - a model that will not terminate")
    print()

# McNemar on the matched items: only the disagreements carry information.
b_only = [q for q in shared if not A[q]["correct"] and B[q]["correct"]]
a_only = [q for q in shared if A[q]["correct"] and not B[q]["correct"]]
n01, n10 = len(b_only), len(a_only)
print(f"matched disagreements: {b_name} only {n01} | {a_name} only {n10}")

if n01 + n10:
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    print(f"McNemar exact (two-sided): p = {p:.3f}")
    if p > 0.05:
        print("  -> not distinguishable at this sample size")
else:
    print("no disagreements")

# How many of the items one model lost were lost to the cap rather than to error?
lost_trunc = sum(1 for q in a_only if kind(B[q]) == "truncated")
print(f"\nof the {n10} items {a_name} won, {lost_trunc} were {b_name} TRUNCATIONS, "
      f"{n10 - lost_trunc} genuine wrong answers")
