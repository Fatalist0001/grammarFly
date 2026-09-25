import sys
import time

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.training import Trainer
from grammar.ab import ABGrammar

SEED = 1


def make_probe(seed):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    return graph, ia, ib, acc, rej, org_a, org_b


def half_features(pipe, t0, t1):
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    if len(t) == 0:
        return np.zeros(8)
    mid = (t0 / ms + t1 / ms) / 2
    sel_early = (t >= t0 / ms) & (t < mid)
    sel_late = (t >= mid) & (t <= t1 / ms)

    def h(sel, idx):
        return np.isin(i[sel], idx).sum()

    n_acc, n_rej = len(pipe.out_acc), len(pipe.out_rej)
    n_a, n_b = len(pipe.in_a), len(pipe.in_b)
    tt = np.unique(t)
    t0s, t1s = t0 / ms, t1 / ms
    early_span = max(tt[(tt >= t0s) & (tt < mid)].max(initial=t0s) - t0s, 0.1) / 1e3
    late_span = max(t1s - tt[(tt >= mid) & (tt <= t1s)].min(initial=t1s), 0.1) / 1e3
    return np.array([
        h(sel_early, pipe.out_acc) / n_acc / early_span,
        h(sel_late, pipe.out_acc) / n_acc / late_span,
        h(sel_early, pipe.out_rej) / n_rej / early_span,
        h(sel_late, pipe.out_rej) / n_rej / late_span,
        h(sel_early, pipe.in_a) / n_a / early_span,
        h(sel_late, pipe.in_a) / n_a / late_span,
        h(sel_early, pipe.in_b) / n_b / early_span,
        h(sel_late, pipe.in_b) / n_b / late_span,
    ])


def collect_features(pipe, samples):
    F, y = [], []
    for w, l in samples:
        pipe.reset_state()
        t0, t1 = pipe.present_word(w)
        F.append(half_features(pipe, t0, t1))
        y.append(l)
    return np.array(F), np.array(y)


def auroc(score, y):
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(score)
    srt = y[order]
    n_pos = (srt == 1).sum()
    n_neg = (srt == 0).sum()
    ranks = np.arange(1, len(srt) + 1)
    pos_ranks = ranks[srt == 1]
    return (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bal_acc(score, y):
    thr = 0.0
    pred = score > thr
    pos = y == 1
    neg = y == 0
    return 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())


CHANNEL_SCORES = {
    "late_early": lambda F: (F[:, 1] + F[:, 3]) - (F[:, 0] + F[:, 2]),
    "acc_L-acc_E": lambda F: F[:, 1] - F[:, 0],
    "rej_L-rej_E": lambda F: F[:, 3] - F[:, 2],
    "mean_rate": lambda F: (F[:, 0] + F[:, 1]) - (F[:, 2] + F[:, 3]),
}

TRAIN = [('A', 0), ('B', 0), ('AB', 1), ('AAB', 0), ('ABB', 0),
         ('ABAB', 0), ('AAABB', 0), ('AABB', 1), ('AAABBB', 1)]


def balanced_eval():
    g = ABGrammar()
    pos = [(g.positive(n), True) for n in g.TRAIN_NS + g.TEST_NS]
    neg_by_len = {}
    for w in g.all_negatives((1, 2, 3, 4, 5, 6, 7, 8)):
        neg_by_len.setdefault(len(w), []).append((w, False))
    rng = np.random.RandomState(7)
    neg = []
    for L, pool in sorted(neg_by_len.items()):
        idx = rng.choice(len(pool), min(200, len(pool)), replace=False)
        neg.extend(pool[int(i)] for i in idx)
    return pos + neg


def report(pipe, tag):
    samples = balanced_eval()
    F, y = collect_features(pipe, samples)
    line = [f"{tag:7s} (pos={y.sum()})"]
    for nm, fn in CHANNEL_SCORES.items():
        s = fn(F)
        line.append(f"{nm}: auroc={auroc(s, y):.2f} bacc={bal_acc(s, y):.2f}")
    print("  " + "  ".join(line), flush=True)


def main():
    print("balanced-eval set: classic train 9-words for fit; eval = all positives "
          "n=1..6 + up to 200 neg per length 1..8", flush=True)
    report_style = "score: auroc over whole balanced set, bacc at threshold 0"

    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    start_scope()
    np.random.seed(SEED)
    graph2, ia2, ib2, acc2, rej2, org_a2, org_b2 = make_probe(SEED)
    pipe = Pipeline(graph2, org_a2, org_b2, ia2, ib2, acc2, rej2,
                    0.8 * nA, 0.001 * nA, plastic=True,
                    kc_mbon_plastic=True, kc_mbon_stdp='single',
                    kc_scope='mbon', tau_el=50 * ms,
                    a_plus=0.02, a_minus=0.01)
    trainer = Trainer(pipe, eta=0.001, w_max_scale=1.1)

    report(pipe, "base")
    for ep in range(1, 21):
        t0 = time.time()
        trainer.fit(TRAIN, epochs=1)
        nsp = len(trainer.pipe.mon_brain.t)
        dt = time.time() - t0
        if ep % 5 == 0 or ep == 1:
            report(pipe, f"ep{ep:02d}")
            print(f"    (ep {ep}: dt={dt:.1f}s spikes={nsp})", flush=True)
        elif ep % 2 == 0:
            print(f"    (ep {ep}: dt={dt:.1f}s spikes={nsp})", flush=True)
        if nsp > 6_000_000 or dt > 400:
            print(f"  ABORT at ep {ep}: pathological activity", flush=True)
            break


if __name__ == "__main__":
    main()