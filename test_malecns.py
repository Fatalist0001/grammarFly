import numpy as np
from brian2 import Network, SpikeMonitor, ms, nA, start_scope

from brain.malecns import MaleCNSGraph

BASE = "data/malecns_mbRlobes"


def test_metadata():
    g = MaleCNSGraph(BASE)
    ids = set(int(b) for b in g.body_ids)
    assert len(ids) == g.n_neurons == len(g.idx)
    assert set(g.pairs["bodyId_pre"]) | set(g.pairs["bodyId_post"]) <= ids
    assert (g.pairs["weight"] >= 1).all()
    assert len(g.node_rois()) == g.n_neurons
    print(f"neurons={g.n_neurons} pairs={g.n_pairs}")


def test_build_and_run_without_learning():
    start_scope()
    np.random.seed(1)
    g = MaleCNSGraph(BASE)
    neurons, syn = g.build()
    spikes = SpikeMonitor(neurons)
    net = Network(neurons, syn, spikes)

    net.run(20 * ms)
    assert len(spikes.t) == 0

    neurons.I[::200] = 2 * nA
    net.run(50 * ms)
    assert len(spikes.t) > 0
    print(f"stimulus spikes={len(spikes.t)}")


if __name__ == "__main__":
    test_metadata()
    test_build_and_run_without_learning()
    print("OK: MaleCNS -> Brian2 converter works without learning")