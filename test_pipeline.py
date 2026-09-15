import numpy as np
from brian2 import Network, SpikeMonitor, Hz, ms, nA, start_scope

from organs.sensory import SensoryOrgan
from organs.readout import Readout
from brain.lif import make_lif, make_synapse

N_ORGAN = 4
N_BRAIN = 30
RATE = 100 * Hz
WINDOW = 50 * ms
W_IN = 0.2 * nA
W_REC = 0.05 * nA
W_OUT = 0.25 * nA


def build():
    a_organ = SensoryOrgan("A", N_ORGAN, RATE)
    b_organ = SensoryOrgan("B", N_ORGAN, RATE)
    brain = make_lif(N_BRAIN, name="brain")
    out_a = make_lif(4, name="out_a")
    out_b = make_lif(4, name="out_b")

    s_a = make_synapse(a_organ.source, brain, w=W_IN, p=0.5, name="s_a")
    s_b = make_synapse(b_organ.source, brain, w=W_IN, p=0.5, name="s_b")
    s_rec = make_synapse(brain, brain, w=W_REC, p=0.1, name="s_rec")
    s_out_a = make_synapse(brain, out_a, w=W_OUT, p=0.3, name="s_out_a")
    s_out_b = make_synapse(brain, out_b, w=W_OUT, p=0.3, name="s_out_b")

    readout = Readout(out_a, out_b)
    brain_spikes = SpikeMonitor(brain)

    net = Network(
        a_organ.source, b_organ.source,
        a_organ.spikes, b_organ.spikes,
        brain, out_a, out_b,
        s_a, s_b, s_rec, s_out_a, s_out_b,
        readout.accept_spikes, readout.reject_spikes,
        brain_spikes,
    )
    return net, a_organ, b_organ, readout, brain_spikes


def present(net, organ, window):
    t0 = net.t
    organ.source.rates = organ.rate
    net.run(window)
    organ.source.rates = 0 * Hz
    return t0, net.t


def run_trial(sequence):
    start_scope()
    np.random.seed(0)
    net, a_organ, b_organ, readout, brain_spikes = build()
    organs = {"A": a_organ, "B": b_organ}
    for symbol in sequence:
        present(net, organs[symbol], WINDOW)
    t0, t1 = 0 * ms, net.t
    return readout.read_output(t0, t1), len(brain_spikes.t), (
        len(readout.accept_spikes.t), len(readout.reject_spikes.t)
    )


def test_empty_trial():
    result, brain_n, out = run_trial("")
    assert result == "REJECT"
    assert brain_n == 0
    assert out == (0, 0)


def test_symbol_to_readout_chain():
    for sequence in ["A", "B", "AB", "AABB"]:
        result, brain_n, out = run_trial(sequence)
        assert brain_n > 0, sequence
        assert sum(out) > 0, sequence
        assert result in ("ACCEPT", "REJECT"), sequence
        print(f"{sequence!r}: readout={result} brain_spikes={brain_n} out={out}")


if __name__ == "__main__":
    test_empty_trial()
    test_symbol_to_readout_chain()
    print("OK: symbol -> organ -> LIF neurons -> readout pipeline works")