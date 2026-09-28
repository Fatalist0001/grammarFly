import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.parallel import ParallelTrainer

SEED = 1
N_PROC = 6


def build_pipe(w_path):
    start_scope()
    np.random.seed(SEED)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=SEED)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=True,
                    kc_mbon_plastic=True, kc_mbon_stdp='order',
                    kc_scope='mbon', tau_el=60 * ms,
                    tau_pre=60 * ms, tau_post=60 * ms,
                    tau_slow=300 * ms,
                    a_plus=0.02, a_minus=0.01,
                    a_plus_slow=0.06, a_minus_slow=0.02)
    if w_path:
        syn = getattr(pipe, "kc_mbon_syn", None)
        syn.w = np.load(w_path)["w"] * nA
    return pipe


def main():
    w_path = sys.argv[1] if len(sys.argv) > 1 else None
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3]

    from grammar.ab import ABGrammar
    words = np.array(list(ABGrammar.all_words(length)))
    probe = build_pipe(w_path)

    mats = []
    for seed in seeds:
        pt = ParallelTrainer(probe, seed=seed, n_proc=N_PROC)
        pt.set_weights(np.asarray(getattr(probe, "kc_mbon_syn", None).w / nA))
        M, NS = pt.metrics(words)
        pt.close()
        diags = (M[:, 3] - M[:, 2]) - (M[:, 1] - M[:, 0])
        mats.append(diags)
        pos_mask = np.array([ABGrammar.is_positive(w) for w in words])
        neg_mask = ~pos_mask
        print(f"seed={seed}: pos={diags[pos_mask].max():+.2f} "
              f"top_neg={diags[neg_mask].max():+.2f}", flush=True)

    mats = np.array(mats)
    mean_diag = mats.mean(axis=0)
    std_diag = mats.std(axis=0)
    order = np.argsort(-mean_diag)
    pos_mask = np.array([ABGrammar.is_positive(w) for w in words])
    pos_idx = int(np.flatnonzero(pos_mask)[0])
    print(f"== len {length} (mean over {seeds}) ==", flush=True)
    for rank in order[:10]:
        w = words[rank]
        mark = "POS" if ABGrammar.is_positive(w) else ("rev" if ABGrammar.has_ba_reversal(w) else "    ")
        print(f"  {w:8s} mean={mean_diag[rank]:+8.2f} std={std_diag[rank]:+5.2f} {mark}", flush=True)
    neg_mask = ~pos_mask
    top_neg = mean_diag[neg_mask].max()
    pos_diag = mean_diag[pos_idx]
    print(f"  pos={words[pos_idx]} pos_mean={pos_diag:+.2f} "
          f"top_neg={top_neg:+.2f} margin={pos_diag - top_neg:+.2f} "
          f"auroc={(mean_diag[pos_idx] > mean_diag[neg_mask]).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()