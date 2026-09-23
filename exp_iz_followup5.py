import sys
sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, mV, nA, start_scope

from organs.sensory import SensoryOrgan
from brain.malecns import MaleCNSGraph
from brain.pipeline import Pipeline
from brain.lif import make_izhikevich

WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA", "AABABB", "ABBAB"]
PAUSE = 300 * ms
WINDOW_BIN = (100, 200)
K = 8
W_IN = 0.8 * nA
w_scale = 0.002 * nA


def build(seed):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan("A", 4, 100 * Hz)
    org_b = SensoryOrgan("B", 4, 100 * Hz)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    w_in=W_IN, w_scale=w_scale,
                    plastic=False, neuron_fn=make_izhikevich,
                    preset=None, a=0.02, b=0.2, d=2.0 * mV)
    return pipe, np.concatenate([acc, rej]), -65 * mV, 0.2


def run_once(pipe, mbon, c, b, word):
    pipe.brain.v = c
    pipe.brain.u = b * c
    pipe.brain.I = 0 * 1e-12 * nA
    t0, t1 = pipe.present_word(word, delay=PAUSE)
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    end = t1 / ms
    w0, w1 = end - PAUSE / ms + WINDOW_BIN[0], end - PAUSE / ms + WINDOW_BIN[1]
    sel = (t >= w0) & (t <= w1)
    it = i[sel]
    return np.array([np.isin(it, nid).sum() for nid in mbon]), len(t)


def canon_intra(fpA):
    m = np.mean(fpA, 0)
    return float(np.mean([np.abs(f - m).sum() for f in fpA]))


def canon_inter(fpA, fpB):
    return float(np.abs(np.mean(fpA, 0) - np.mean(fpB, 0)).sum())


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    acc, rej = graph.mbon_indices()
    mbon = np.concatenate([acc, rej])
    # fixed KC indices, vary only Poisson stream (canonical protocol)
    pipe, _, c, b = build(0)
    prof = {w: [] for w in WORDS}
    tot = {w: [] for w in WORDS}
    for k in range(K):
        np.random.seed(100 + k)
        for word in WORDS:
            v, t = run_once(pipe, mbon, c, b, word)
            prof[word].append(v)
            tot[word].append(t)
    # also verify across fresh builds (varying KC)
    prof2 = {w: [] for w in WORDS}
    for k in range(K):
        pipe, mbon2, c, b = build(2000 + k)
        for word in WORDS:
            v, t = run_once(pipe, mbon2, c, b, word)
            prof2[word].append(v)

    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]
    print("=== d=2 canonical (fixed KC, K=8 poisson draws) ===")
    for (wa, wb) in pairs:
        A = np.array(prof[wa]); B = np.array(prof[wb])
        intraA = canon_intra(A); intraB = canon_intra(B)
        inter = canon_inter(A, B)
        ratioA = intraA / (intraA + inter + 1e-12)
        ratioB = intraB / (intraB + inter + 1e-12)
        print(f"  {wa}/{wb}: d_intra={intraA:7.1f}/{intraB:7.1f} "
              f"d_inter={inter:7.1f} ratio={ratioA:.3f}/{ratioB:.3f} "
              f"totA={np.mean(tot[wa]):.0f} totB={np.mean(tot[wb]):.0f}")
    words4 = ["ABAB", "BAAB", "ABBA", "AABB", "AB", "BA", "AAB", "ABB"]
    print("=== d=2 cross-build (varying KC, K=8) all words ===")
    for (wa, wb) in pairs:
        A = np.array(prof2[wa]); B = np.array(prof2[wb])
        inter = canon_inter(A, B)
        intraA = canon_intra(A); intraB = canon_intra(B)
        print(f"  {wa}/{wb}: intra={intraA:.1f}/{intraB:.1f} inter={inter:.1f} "
              f"ratio={intraA/(intraA+inter+1e-12):.3f}")

    print("=== d=2 within-AB/BA scatter (canon, per realization) ===")
    A = np.array(prof["AB"]); B = np.array(prof["BA"])
    for k in range(K):
        d_intraAB = np.abs(A[k] - A.mean(0)).sum()
        print(f"  AB k{k}: tot={tot['AB'][k]:7.0f} |A-Abar|={d_intraAB:8.1f}")
    for k in range(K):
        print(f"  BA k{k}: tot={tot['BA'][k]:7.0f}")


if __name__ == "__main__":
    main()