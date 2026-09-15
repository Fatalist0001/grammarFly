import numpy as np
from brian2 import Hz, nA, start_scope

from organs.sensory import SensoryOrgan
from brain.malecns import MaleCNSGraph
from brain.pipeline import Pipeline
from brain.training import Trainer

BASE = "data/malecns_mbRlobes"
N_KC = 100
RATE = 100 * Hz
W_IN = 0.8 * nA
W_SCALE = 0.001 * nA

TRAIN = [
    ("AB", True), ("AABB", True), ("AAABBB", True),
    ("A", False), ("B", False), ("AAB", False),
    ("ABB", False), ("ABAB", False), ("AAABB", False),
]


def build():
    start_scope()
    np.random.seed(7)
    graph = MaleCNSGraph(BASE)
    a, b = graph.kc_indices(N_KC, seed=7)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan("A", 4, RATE)
    org_b = SensoryOrgan("B", 4, RATE)
    pipe = Pipeline(graph, org_a, org_b, a, b, acc, rej, W_IN, W_SCALE,
                    plastic=True)
    return pipe


def test_training_loop():
    pipe = build()
    trainer = Trainer(pipe, eta=0.02)

    base = trainer.evaluate(TRAIN)
    base_acc = np.mean([(r == "ACCEPT") == bool(l) for _, l, r in base])
    print(f"baseline train_acc (untrained): {base_acc:.3f}")

    w_before = np.asarray(trainer.syn.w / nA).copy()
    history = trainer.fit(TRAIN, epochs=1)
    w_after = np.asarray(trainer.syn.w / nA)

    assert len(history) == 1
    assert 0.0 <= history[0] <= 1.0
    assert np.allclose(np.asarray(trainer.syn.elig), 0)
    assert (w_after > w_before).any() or (w_after < w_before).any()
    assert np.all(w_after >= trainer.w_min - 1e-12)
    assert np.all(w_after <= trainer.w_max + 1e-12)
    print(f"train_acc after 1 epoch: {history[0]:.3f}")

    trainer.reset_weights_to_initial()
    w_reset = np.asarray(trainer.syn.w / nA)
    assert np.allclose(w_reset, w_before)


if __name__ == "__main__":
    test_training_loop()
    print("OK: training loop verified")