import numpy as np
from brian2 import Hz, ms, nA, start_scope

from organs.sensory import SensoryOrgan
from brain.malecns import MaleCNSGraph
from brain.pipeline import Pipeline

BASE = "data/malecns_mbRlobes"
WINDOW = 50 * ms
N_KC = 100
RATE = 100 * Hz
W_IN = 0.8 * nA
MEAN_RATE_CAP = 100.0
MAX_RATE_CAP = 250.0

SEQUENCES = ["A", "B", "AB", "AABB"]


def build_pipeline(w_scale):
    start_scope()
    np.random.seed(7)
    graph = MaleCNSGraph(BASE)
    a, b = graph.kc_indices(N_KC, seed=7)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan("A", 4, RATE)
    org_b = SensoryOrgan("B", 4, RATE)
    return Pipeline(graph, org_a, org_b, a, b, acc, rej, W_IN, w_scale)


def run_sequence(pipe, sequence):
    t0 = pipe.net.t
    organs = {"A": pipe.organ_a, "B": pipe.organ_b}
    for symbol in sequence:
        org = organs[symbol]
        org.source.rates = org.rate
        pipe.net.run(WINDOW)
        org.source.rates = 0 * Hz
    return t0, pipe.net.t


def rates(pipe, t0, t1):
    return {
        "kc_a": pipe.rate(pipe.in_a, t0, t1),
        "kc_b": pipe.rate(pipe.in_b, t0, t1),
        "mbon_acc": pipe.rate(pipe.out_acc, t0, t1),
        "mbon_rej": pipe.rate(pipe.out_rej, t0, t1),
        "brain": pipe.rate(np.arange(len(pipe.brain)), t0, t1),
    }


def calibrate(pipe):
    candidates = [c * nA for c in (0.001, 0.002, 0.004, 0.008, 0.016)]
    measured = []
    for ws in candidates:
        pipe.reset_weights(ws)
        t0, t1 = run_sequence(pipe, "AB")
        r = rates(pipe, t0, t1)
        measured.append((ws, r["mbon_acc"] + r["mbon_rej"], r["brain"]))
        print(f"  w_scale={ws!r}  mbon={r['mbon_acc'] + r['mbon_rej']:.1f} Hz  "
              f"brain={r['brain']:.1f} Hz")
    bounded = [m for m in measured if m[2] < MEAN_RATE_CAP]
    if bounded:
        return max(bounded, key=lambda m: m[1])[0]
    return min(measured, key=lambda m: m[2])[0]


def test_sequence_sets():
    pipe = build_pipeline(0.002 * nA)
    assert len(pipe.mon_brain.t) == 0  # no spontaneous activity before input
    best = calibrate(pipe)
    print(f"chosen w_scale = {best!r}")
    pipe.reset_weights(best)

    signatures = {}
    for seq in SEQUENCES:
        t0, t1 = run_sequence(pipe, seq)
        r = rates(pipe, t0, t1)
        assert r["brain"] > 0.0, f"activity died for {seq}"
        assert r["brain"] < MAX_RATE_CAP, f"saturated for {seq}: {r['brain']}"
        signatures[seq] = tuple(round(r[k], 3) for k in ("kc_a", "kc_b", "mbon_acc", "mbon_rej"))
        assert pipe.read_output(t0, t1) in ("ACCEPT", "REJECT")
        print(f"{seq!r}: {r}")
    assert len(set(signatures.values())) == len(SEQUENCES)

    t0, t1 = run_sequence(pipe, "AABB")
    mid = (t0 + t1) / 2
    first = rates(pipe, t0, mid)
    second = rates(pipe, mid, t1)
    assert first["kc_a"] > first["kc_b"] and second["kc_b"] > second["kc_a"]


if __name__ == "__main__":
    test_sequence_sets()
    print("OK: MaleCNS subgraph propagates sequence signal without dying/saturating")