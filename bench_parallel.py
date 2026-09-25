import sys
import time

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, nA, ms

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.parallel import collect_features_parallel
from grammar.ab import ABGrammar

SEED = 1


def make_probe(seed):
    from brian2 import start_scope
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


def collect_serial(pipe, samples):
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
    s = score.copy()
    order = np.argsort(s)
    srt = y[order]
    n_pos = (srt == 1).sum()
    n_neg = (srt == 0).sum()
    ranks = np.arange(1, len(srt) + 1)
    pos_ranks = ranks[srt == 1]
    return (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def build_samples(n_neg_per_len=60):
    g = ABGrammar()
    pos = [(g.positive(n), True) for n in g.TRAIN_NS + g.TEST_NS]
    by_len = {}
    for w in g.all_negatives((1, 2, 3, 4, 5, 6)):
        by_len.setdefault(len(w), []).append((w, False))
    rng = np.random.RandomState(3)
    neg = []
    for L, pool in sorted(by_len.items()):
        idx = rng.choice(len(pool), min(n_neg_per_len, len(pool)), replace=False)
        neg.extend(pool[int(i)] for i in idx)
    return pos + neg


def main():
    samples = build_samples()
    n = len(samples)
    n_pos = sum(l for _, l in samples)
    print(f"samples={n} (pos={n_pos})", flush=True)

    # one shared pipe; serial on it, parallel re-builds copies in workers
    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=False)

    t0 = time.time()
    Fs, ys = collect_serial(pipe, samples)
    dt_serial = time.time() - t0
    print(f"serial:   {dt_serial:.1f}s   late_early AUROC="
          f"{auroc((Fs[:,1]+Fs[:,3])-(Fs[:,0]+Fs[:,2]), ys):.3f}", flush=True)

    for n_proc in (2, 4, 6):
        t0 = time.time()
        Fp, yp = collect_features_parallel(pipe, samples, seed=SEED, n_proc=n_proc)
        dt = time.time() - t0
        same_labels = np.array_equal(ys, yp)
        feats_close = np.allclose(Fs, Fp, atol=1e-12) if same_labels else False
        a_p = auroc((Fp[:,1]+Fp[:,3])-(Fp[:,0]+Fp[:,2]), yp)
        print(f"parallel {n_proc}): {dt:.1f}s  speedup={dt_serial/dt:.2f}x  "
              f"AUROC={a_p:.3f}  labels_same={same_labels} feats_exact={feats_close}",
              flush=True)


if __name__ == "__main__":
    main()