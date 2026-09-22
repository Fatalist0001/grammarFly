import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from brian2 import Hz, nA, ms, start_scope

WORDS = ['AB', 'AABB', 'AAABBB', 'A', 'B', 'AAB', 'ABB', 'ABAB', 'AAABB']
LABELS = [1, 1, 1, 0, 0, 0, 0, 0, 0]
OUT = os.path.join(os.path.dirname(__file__), "malecns_mbRlobes", "readout_split.csv")


def main():
    from brain.malecns import MaleCNSGraph
    from organs.sensory import SensoryOrgan
    from brain.pipeline import Pipeline

    graph = MaleCNSGraph(os.path.join(os.path.dirname(__file__), "malecns_mbRlobes"))
    mbons = [(i, t) for i, t in enumerate(graph.types) if t.startswith("MBON")]

    rates = {w: {} for w in WORDS}
    for w in WORDS:
        start_scope()
        np.random.seed(0)
        a, b = graph.kc_indices(100, seed=0)
        acc_dummy, rej_dummy = graph.mbon_indices()
        org_a = SensoryOrgan('A', 4, 100 * Hz)
        org_b = SensoryOrgan('B', 4, 100 * Hz)
        pipe = Pipeline(graph, org_a, org_b, a, b, acc_dummy, rej_dummy,
                        0.8 * nA, 0.001 * nA, plastic=False)
        pipe.reset_state()
        t0, t1 = pipe.present_word(w)
        t = np.asarray(pipe.mon_brain.t / ms)
        i = pipe.mon_brain.i
        sel = (t >= t0 / ms) & (t <= t1 / ms)
        dur = (t1 - t0) / ms * 1e-3
        for idx, mbon_type in mbons:
            rates[w][idx] = np.sum(sel & (i == idx)) / dur if dur > 0 else 0.0

    rows = []
    for idx, mbon_type in mbons:
        pos = [rates[w][idx] for w, lab in zip(WORDS, LABELS) if lab == 1]
        neg = [rates[w][idx] for w, lab in zip(WORDS, LABELS) if lab == 0]
        m_pos = float(np.mean(pos))
        m_neg = float(np.mean(neg))
        row = {"bodyId": int(graph.body_ids[idx]), "type": mbon_type,
               "idx": int(idx), "rate_pos": m_pos, "rate_neg": m_neg,
               "diff": m_pos - m_neg}
        rows.append(row)
    df = pd.DataFrame(rows)

    pos_sel = df[df["diff"] > 0].sort_values("diff", ascending=False)
    neg_sel = df[df["diff"] < 0].sort_values("diff", ascending=True)

    if len(pos_sel) >= 2 and len(neg_sel) >= 2:
        n = min(len(pos_sel), len(neg_sel))
        accept = pos_sel.head(n)
        reject = neg_sel.head(n)
        balanced = True
    else:
        mid = len(df) // 2
        accept = df.iloc[:mid]
        reject = df.iloc[mid:]
        balanced = False

    out = pd.DataFrame({
        "bodyId": list(accept["bodyId"]) + list(reject["bodyId"]),
        "type": list(accept["type"]) + list(reject["type"]),
        "channel": ["ACCEPT"] * len(accept) + ["REJECT"] * len(reject),
        "diff": list(accept["diff"]) + list(reject["diff"]),
    })
    out.to_csv(OUT, index=False)
    print(f"saved {OUT}: ACCEPT={len(accept)}, REJECT={len(reject)}, balanced={balanced}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()