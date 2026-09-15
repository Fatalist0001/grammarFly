import os
import numpy as np
from brian2 import Hz, ms, nA, start_scope

from organs.sensory import SensoryOrgan
from brain.malecns import MaleCNSGraph
from brain.pipeline import Pipeline
from brain.training import Trainer
from grammar.ab import ABGrammar

BASE = "data/malecns_mbRlobes"
N_KC = 100
RATE = 100 * Hz
W_IN = 0.8 * nA
W_SCALE = 0.001 * nA
SEEDS = [0, 1]
EPOCHS = 20
ETA = 0.02
W_MAX_SCALE = 2.0
OUT_DIR = "results"

TRAIN = [
    ("AB", True), ("AABB", True), ("AAABBB", True),
    ("A", False), ("B", False), ("AAB", False),
    ("ABB", False), ("ABAB", False), ("AAABB", False),
]


def build(seed):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph(BASE)
    a, b = graph.kc_indices(N_KC, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan("A", 4, RATE)
    org_b = SensoryOrgan("B", 4, RATE)
    return Pipeline(graph, org_a, org_b, a, b, acc, rej, W_IN, W_SCALE,
                    plastic=True)


def make_test():
    gram = ABGrammar()
    pos = [(gram.positive(n), True) for n in gram.TEST_NS]
    rng = np.random.default_rng(42)
    all_neg = [(w, False)
               for w in gram.all_negatives(range(7, 11))]
    idx = rng.choice(len(all_neg), size=30, replace=False)
    neg = [all_neg[i] for i in sorted(idx)]
    return pos + neg


def eval_table(samples, label=""):
    preds = [(w, l, r) for w, l, r in samples]
    correct = sum((r == "ACCEPT") == l for _, l, r in preds)
    total = len(preds)
    acc = correct / total
    pos = [w for w, l, _ in preds if l]
    neg = [w for w, l, _ in preds if not l]
    pos_correct = sum((r == "ACCEPT") for _, l, r in preds if l)
    neg_correct = sum((r == "REJECT") for _, l, r in preds if not l)
    acc_pos = pos_correct / len(pos) if pos else float("nan")
    acc_neg = neg_correct / len(neg) if neg else float("nan")
    per_n = {}
    for n in ABGrammar.TEST_NS:
        words = [w for w, _, r in preds if w == ABGrammar.positive(n)]
        labs = [l for w, l, _ in preds if w == ABGrammar.positive(n)]
        res = [r for w, l, r in preds if w == ABGrammar.positive(n)]
        correct_n = sum((rr == "ACCEPT") == ll for rr, ll in zip(res, labs))
        per_n[n] = correct_n / len(labs) if labs else float("nan")
    print(f"  {label}: total={acc:.3f} pos_acc={acc_pos:.3f} "
          f"neg_acc={acc_neg:.3f} per_n={per_n}")
    return {"acc": acc, "acc_pos": acc_pos, "acc_neg": acc_neg, "per_n": per_n}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    test_samples = make_test()

    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        pipe = build(seed)
        trainer = Trainer(pipe, eta=ETA, w_max_scale=W_MAX_SCALE)
        base = trainer.evaluate(TRAIN + test_samples)
        base_m = eval_table(base, label="baseline")
        history = trainer.fit(TRAIN, epochs=EPOCHS, verbose=True)
        final = trainer.evaluate(TRAIN + test_samples)
        final_m = eval_table(final, label="final ")
        results.append({
            "seed": seed,
            "history": history,
            "base": base_m,
            "final": final_m,
        })
        np.save(f"{OUT_DIR}/history_seed{seed}.npy", np.array(history))

    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        for r in results:
            f.write(f"seed {r['seed']}\n")
            f.write(f"  base_acc={r['base']['acc']:.3f}\n")
            f.write(f"  final_acc={r['final']['acc']:.3f}\n")
            f.write(f"  per_n={r['final']['per_n']}\n\n")
    print("\nsaved results/summary.txt and per-seed history files")