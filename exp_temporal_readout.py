import sys
import time

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.training import Trainer

TRAIN = [('A', 0), ('B', 0), ('AB', 1), ('AAB', 0), ('ABB', 0),
         ('ABAB', 0), ('AAABB', 0), ('AABB', 1), ('AAABBB', 1)]
TEST = [('AAAB', 1), ('BAA', 0), ('AABBB', 0), ('AAAABBBB', 1),
        ('ABBA', 0), ('AAAA', 0), ('BBBB', 0), ('AAAAB', 0),
        ('ABBB', 0), ('AAAAABBBBB', 1), ('BA', 0), ('AABBAA', 0)]


def make_probe(seed):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    return graph, ia, ib, acc, rej, org_a, org_b


def fit_readout(seed):
    graph, ia, ib, acc, rej, org_a, org_b = make_probe(seed)
    probe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                     0.8 * nA, 0.001 * nA, plastic=False)
    F = np.zeros((len(TRAIN), 2 * probe.brain.N))
    y = np.array([l for _, l in TRAIN])
    for wi, (w, _) in enumerate(TRAIN):
        probe.reset_state()
        t0, t1 = probe.present_word(w)
        F[wi] = probe._temporal_features(t0, t1)
    mu1 = F[y == 1].mean(0)
    mu0 = F[y == 0].mean(0)
    Sw = np.cov((F[y == 1] - mu1).T) + np.cov((F[y == 0] - mu0).T)
    W = np.linalg.solve(Sw + 1e-3 * np.eye(F.shape[1]), mu1 - mu0)
    s = F @ W
    thr = (s[y == 1].mean() + s[y == 0].mean()) / 2
    return W, thr


def run_seed(seed, epochs=30, eta=0.01, w_max_scale=1.5, eval_every=5,
             scope='mbon', a_plus=0.02, a_minus=0.01):
    W, thr = fit_readout(seed)
    graph, ia, ib, acc, rej, org_a, org_b = make_probe(seed)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=True,
                    kc_mbon_plastic=True, kc_mbon_stdp='single',
                    kc_scope=scope,
                    tau_el=50 * ms, a_plus=a_plus, a_minus=a_minus,
                    readout_w=W, readout_thr=thr)
    trainer = Trainer(pipe, eta=eta, w_max_scale=w_max_scale)

    def acc_over(samples):
        r = trainer.evaluate(samples)
        return sum((x == 'ACCEPT') == bool(l) for _, l, x in r) / len(r)

    base_tr, base_te = acc_over(TRAIN), acc_over(TEST)
    best = {'te': 0.0, 'ep': 0}
    hist_tr, hist_te = [], []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        trainer.fit(TRAIN, epochs=1)
        nsp = len(trainer.pipe.mon_brain.t)
        dt = time.time() - t0
        if ep % eval_every == 0:
            at, ae = acc_over(TRAIN), acc_over(TEST)
            hist_tr.append(at)
            hist_te.append(ae)
            if ae > best['te']:
                best = {'te': ae, 'ep': ep}
            print(f'  [seed {seed}] ep {ep:3d} dt={dt:5.1f}s spikes={nsp} '
                  f'train={at:.2f} test={ae:.2f}', flush=True)
        elif ep % 3 == 0 or ep == 1:
            print(f'  [seed {seed}] ep {ep:3d} dt={dt:5.1f}s spikes={nsp}',
                  flush=True)
        if nsp > 6_000_000 or dt > 400:
            print(f'  [seed {seed}] ABORT at ep {ep}: pathological activity',
                  flush=True)
            break
    final_te = acc_over(TEST)
    return (base_tr, base_te), (hist_tr, hist_te), final_te, best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1 2')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--eta', type=float, default=0.01)
    ap.add_argument('--wmax', type=float, default=1.5)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--scope', default='kc_out')
    ap.add_argument('--aplus', type=float, default=0.02)
    ap.add_argument('--aminus', type=float, default=0.01)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split()]
    print(f'seed | base_tr base_te | best_te@ep | final_te')
    results = []
    for seed in seeds:
        base, hist, final_te, best = run_seed(
            seed, epochs=args.epochs, eta=args.eta,
            w_max_scale=args.wmax, eval_every=args.eval_every,
            scope=args.scope, a_plus=args.aplus, a_minus=args.aminus)
        tr, te = hist
        print(f'{seed:4d} | {base[0]:6.2f} {base[1]:6.2f} | '
              f'{best["te"]:6.2f}@{best["ep"]:<3d} | final_te={final_te:.2f}')
        results.append((base[0], base[1], best['te'], final_te))
    results = np.array(results)
    print('means: base_tr/te = %.2f/%.2f  best_te = %.2f  final_te = %.2f'
          % (results[:, 0].mean(), results[:, 1].mean(),
             results[:, 2].mean(), results[:, 3].mean()))


if __name__ == "__main__":
    main()