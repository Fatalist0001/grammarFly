import sys
import time

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.training import Trainer
from brain.parallel import ParallelTrainer

SEED = 1
N_PROC = 6

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


def main():
    tau_slow = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
    tau_el = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    eta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    w_max_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.1
    n_epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    a_slow_factor = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    asym = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    channel_only = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    norm = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    profile = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    print(f"order-STDP: tau_slow={tau_slow:.0f}ms tau_el={tau_el:.0f}ms "
          f"eta={eta} wmax={w_max_scale} epochs={n_epochs} "
          f"a_slow_factor={a_slow_factor} asym_plus={asym} "
          f"channel_only={channel_only} norm={norm} profile={profile} "
          f"n_proc={N_PROC}", flush=True)
    a_plus_slow = 0.02 * a_slow_factor * asym
    a_minus_slow = 0.01 * a_slow_factor / asym

    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=True,
                    kc_mbon_plastic=True, kc_mbon_stdp='order',
                    kc_scope='mbon', tau_el=tau_el * ms,
                    tau_pre=tau_el * ms, tau_post=tau_el * ms,
                    tau_slow=tau_slow * ms,
                    a_plus=0.02, a_minus=0.01,
                    a_plus_slow=a_plus_slow, a_minus_slow=a_minus_slow)
    trainer = Trainer(pipe, eta=eta, w_max_scale=w_max_scale,
                      channel_reward_only=bool(channel_only),
                      reward_norm="word" if norm else None,
                      reward_mode="profile" if profile else "class")
    syn = trainer.syn
    pt = ParallelTrainer(pipe, seed=SEED, n_proc=N_PROC)
    pt.set_weights(np.asarray(syn.w / nA))

    def run_pairs(tag):
        rows = []
        total_spikes = 0
        for pos, negs in ORDER_PAIRS:
            words = [pos] + negs
            M, NS = pt.metrics(words)
            total_spikes += int(NS.sum())
            got = {}
            for k, w in enumerate(words):
                rejE, rejL, accE, accL = M[k]
                le = (accL + rejL) - (accE + rejE)
                rle = rejL - rejE
                ale = accL - accE
                diag = ale - rle
                got[w] = (le, rle, ale, diag)
            m1 = got[pos][0] - max(got[n][0] for n in negs)
            m2 = got[pos][1] - max(got[n][1] for n in negs)
            m3 = got[pos][2] - max(got[n][2] for n in negs)
            m4 = got[pos][3] - max(got[n][3] for n in negs)
            rows.append((pos, m1, m2, m3, m4))
            print(f"    {tag} len{len(pos):2d} pos={pos}: "
                  f"margin l_e={m1:+.2f} rej={m2:+.2f} acc={m3:+.2f} "
                  f"diag={m4:+.2f} | pos=({got[pos][0]:.1f},{got[pos][3]:.1f}) "
                  f"negs=" + ",".join(f"{got[n][3]:.1f}" for n in negs), flush=True)
        ok = sum(1 for _, _, _, _, m4 in rows if m4 > 0)
        return ok, len(rows), rows, total_spikes

    ok, tot, _, ns = run_pairs("base")
    print(f"  base: diag margin>0 in {ok}/{tot} spikes={ns}", flush=True)
    total_spikes = 0
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        contrib, corrects, ns = pt.fit_epoch(
            TRAIN,
            reward_mode="profile" if profile else "class",
            channel_reward_only=bool(channel_only),
            reward_norm="word" if norm else None)
        total_spikes += int(ns.sum())
        acc_elig = contrib.sum(axis=0)
        w = np.asarray(syn.w / nA) + eta * acc_elig
        np.clip(w, trainer.w_min, trainer.w_max, out=w)
        syn.w = w * nA
        pt.set_weights(np.asarray(syn.w / nA))
        dt = time.time() - t0
        if ep % 5 == 0 or ep == 1:
            ok, tot, rows, ns2 = run_pairs(f"ep{ep:02d}")
            total_spikes += ns2
            print(f"    (ep {ep}: dt={dt:.1f}s spikes={total_spikes})", flush=True)
        if total_spikes > 6_000_000 or dt > 400:
            print(f"  ABORT at ep {ep}: pathological activity", flush=True)
            break
    pt.close()


if __name__ == "__main__":
    main()