import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.training import Trainer

SIMPLE = [('A', 1), ('B', 0)]
ABNBN = [('AB', 1), ('AABB', 1), ('AAABBB', 1), ('A', 0), ('B', 0),
         ('AAB', 0), ('ABB', 0), ('ABAB', 0), ('AAABB', 0)]

CONFIGS = [
    (50 * ms, 0.02, 0.01),
    (50 * ms, 0.05, 0.01),
    (50 * ms, 0.10, 0.01),
    (100 * ms, 0.02, 0.01),
    (100 * ms, 0.05, 0.01),
    (100 * ms, 0.10, 0.01),
    (20 * ms, 0.05, 0.01),
    (20 * ms, 0.10, 0.01),
    (200 * ms, 0.05, 0.01),
]


def run_config(tau_el, a_plus, a_minus, eta, samples, epochs, seed=0):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=True,
                    kc_mbon_plastic=True,
                    tau_el=tau_el, a_plus=a_plus, a_minus=a_minus)
    trainer = Trainer(pipe, eta=eta, w_max_scale=2.0)
    base = trainer.evaluate(samples)
    base_acc = sum((r == 'ACCEPT') == bool(l) for _, l, r in base) / len(base)
    hist = trainer.fit(samples, epochs=epochs)
    final = trainer.evaluate(samples)
    final_acc = sum((r == 'ACCEPT') == bool(l) for _, l, r in final) / len(final)
    return base_acc, final_acc, hist


def main():
    print(f"{'tau_el':>7} {'A+/A-':>7} {'eta':>5} | simple A/B base->final   | A^nB^n base->final")
    for tau_el, a_plus, a_minus in CONFIGS:
        eta = 0.01
        b1, f1, h1 = run_config(tau_el, a_plus, a_minus, eta, SIMPLE, 20)
        b2, f2, h2 = run_config(tau_el, a_plus, a_minus, eta, ABNBN, 20)
        r1 = h1[-1] if h1 else 0
        r2 = h2[-1] if h2 else 0
        print(f"{int(tau_el/ms):>6}ms {a_plus/a_minus:>6.1f}:1 {eta:>5} | "
              f"base={b1:.2f} final={f1:.2f} {(f1-b1):+.2f}      | base={b2:.2f} final={f2:.2f} {(f2-b2):+.2f}")


if __name__ == "__main__":
    main()