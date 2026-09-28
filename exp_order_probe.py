import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.parallel import ParallelTrainer
from grammar.ab import ABGrammar

SEED = 1
N_PROC = 6


def main():
    w_path = sys.argv[1] if len(sys.argv) > 1 else None
    lens = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [8, 10, 12]
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 12

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
    syn = getattr(pipe, "kc_mbon_syn", None)
    if w_path:
        w = np.load(w_path)["w"]
        syn.w = w * nA
    pt = ParallelTrainer(pipe, seed=SEED, n_proc=N_PROC)
    pt.set_weights(np.asarray(syn.w / nA))

    for length in lens:
        words = list(ABGrammar.all_words(length))
        M, NS = pt.metrics(words)
        diags = (M[:, 3] - M[:, 2]) - (M[:, 1] - M[:, 0])
        order = np.argsort(-diags)
        print(f"== len {length}: top diag ==", flush=True)
        for rank in order[:topk]:
            w = words[rank]
            mark = "POS" if ABGrammar.is_positive(w) else ("rev" if ABGrammar.has_ba_reversal(w) else "    ")
            print(f"  {w:8s} diag={diags[rank]:+8.2f} {mark}", flush=True)
        pos_w = ABGrammar.positive(length // 2)
        pos_rank = int(np.flatnonzero(np.array(words) == pos_w)[0])
        pos_diag = diags[pos_rank]
        negm = -np.inf
        negw = None
        for i in order:
            if not ABGrammar.is_positive(words[i]):
                negm, negw = diags[i], words[i]
                break
        print(f"  pos={pos_w} pos_diag={pos_diag:+.2f} rank={int(np.sum(diags > pos_diag))}"
              f" max_neg={negm:+.2f} ({negw})", flush=True)
    pt.close()


if __name__ == "__main__":
    main()