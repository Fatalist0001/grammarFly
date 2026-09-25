import sys
import time

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.training import Trainer

SEED = 1

TRAIN = [('A', 0), ('B', 0), ('AB', 1), ('AAB', 0), ('ABB', 0),
         ('ABAB', 0), ('AAABB', 0), ('AABB', 1), ('AAABBB', 1)]

ORDER_PAIRS = [
    ('AB', ['BA', 'AA']),
    ('AABB', ['ABAB', 'BBAA', 'BAAB']),
    ('AAABBB', ['AABBAA', 'BAAABB', 'ABABAB', 'AAABAB']),
]


def make_probe(seed):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    return graph, ia, ib, acc, rej, org_a, org_b


def channel_score(pipe, word):
    t0, t1 = pipe.present_word(word)
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    if len(t) == 0:
        return 0.0, 0.0, 0.0
    mid = (t0 / ms + t1 / ms) / 2
    sel_e = (t >= t0 / ms) & (t < mid)
    sel_l = (t >= mid) & (t <= t1 / ms)
    n_acc = len(pipe.out_acc)
    rejE = np.isin(i[sel_e], pipe.out_rej).sum() / n_acc
    rejL = np.isin(i[sel_l], pipe.out_rej).sum() / n_acc
    accE = np.isin(i[sel_e], pipe.out_acc).sum() / n_acc
    accL = np.isin(i[sel_l], pipe.out_acc).sum() / n_acc
    return (accL + rejL) - (accE + rejE), rejL - rejE, (accL - accE)


def bool_label(word, pair):
    pos, negs = pair
    if word == pos:
        return True
    return False


def main():
    nneg = sum(len(negs) for _, negs in ORDER_PAIRS)
    print(f"order pairs: 1 pos per {nneg} negs per length {[len(p) for _, p in ORDER_PAIRS]}",
          flush=True)

    start_scope()
    np.random.seed(SEED)
    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=True,
                    kc_mbon_plastic=True, kc_mbon_stdp='single',
                    kc_scope='mbon', tau_el=50 * ms,
                    a_plus=0.02, a_minus=0.01)
    trainer = Trainer(pipe, eta=0.001, w_max_scale=1.1)

    def run_pairs(tag):
        rows = []
        for pos, negs in ORDER_PAIRS:
            words = [pos] + negs
            got = {}
            for w in words:
                pipe.reset_state()
                got[w] = channel_score(pipe, w)
            # margin: score(pos) > max(score(neg)) for each metric
            m1 = got[pos][0] - max(got[n][0] for n in negs)
            m2 = got[pos][1] - max(got[n][1] for n in negs)
            m3 = got[pos][2] - max(got[n][2] for n in negs)
            rows.append((pos, m1, m2, m3))
            print(f"    {tag} len{len(pos):2d} pos={pos}: "
                  f"late_early margin={m1:+.2f} rej_L-E={m2:+.2f} acc_L-E={m3:+.2f} "
                  f"| vals pos=({got[pos][0]:.1f},{got[pos][1]:.1f}) "
                  f"negs=" + ",".join(f"{got[n][0]:.1f}" for n in negs), flush=True)
        ok = sum(1 for _, m1, _, _ in rows if m1 > 0)
        return ok, len(rows)

    ok, tot = run_pairs("base")
    print(f"  base: late_early margin>0 in {ok}/{tot} lengths", flush=True)
    for ep in range(1, 31):
        t0 = time.time()
        trainer.fit(TRAIN, epochs=1)
        dt = time.time() - t0
        if ep % 3 == 0:
            ok, tot = run_pairs(f"ep{ep:02d}")
            print(f"    (ep {ep}: dt={dt:.1f}s spikes="
                  f"{len(trainer.pipe.mon_brain.t)})", flush=True)
        if len(trainer.pipe.mon_brain.t) > 6_000_000 or dt > 400:
            print(f"  ABORT at ep {ep}: pathological activity", flush=True)
            break


if __name__ == "__main__":
    main()