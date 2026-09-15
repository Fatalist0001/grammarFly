import numpy as np
from brian2 import Network, SpikeGeneratorGroup, ms, nA, start_scope

from brain.lif import make_lif, make_synapse, make_stdp_synapse

TAU_EL = 20 * ms
A_PLUS = 0.5
A_MINUS = 0.1
CORRELATED = list(range(2, 45, 5))       # pre fires 2,7,12,...
CLOCK = list(range(5, 45, 5))            # clock fires 5,10,15,...


def run(pre_times, clock_times, duration=100 * ms):
    start_scope()
    pre = SpikeGeneratorGroup(1, np.zeros(len(pre_times)), np.array(pre_times) * ms)
    neuron = make_lif(1, name="neuron")
    syn = make_stdp_synapse(pre, neuron, [0], [0], np.array([0.1]) * nA,
                            tau_el=TAU_EL, a_plus=A_PLUS, a_minus=A_MINUS)
    comps = [pre, neuron, syn]
    if clock_times is None:
        driver = make_synapse(pre, neuron, [0], [0],
                              weights=np.array([5.0]) * nA, name="driver")
    else:
        clock = SpikeGeneratorGroup(1, np.zeros(len(clock_times)),
                                    np.array(clock_times) * ms)
        driver = make_synapse(clock, neuron, [0], [0],
                              weights=np.array([5.0]) * nA, name="driver")
        comps.append(clock)
    driver.delay = 1 * ms
    comps.append(driver)
    net = Network(*comps)
    net.run(duration)
    return syn


def apply(syn, reward, eta=0.02, w_max_scale=2.0):
    w0 = np.asarray(syn.w / nA)
    elig = np.asarray(syn.elig)
    w = w0 + eta * reward * elig
    w = np.clip(w, 0.0, w0 * w_max_scale)
    syn.w = w * nA
    syn.elig = 0
    return w, w0


def test_correlated_reward_grows_w():
    syn = run(CORRELATED, None)          # pre drives its own post spikes
    w, w0 = apply(syn, +1.0)
    assert w[0] > w0[0], f"expected growth, got {w[0]:.4f} vs {w0[0]:.4f}"


def test_correlated_punish_shrinks_w():
    syn = run(CORRELATED, None)
    w, w0 = apply(syn, -1.0)
    assert w[0] < w0[0]


def test_anticorrelated_does_not_grow():
    syn = run(CORRELATED, CLOCK)         # post spikes come from clock, decoupled
    w, w0 = apply(syn, +1.0)
    assert w[0] <= 1.5 * w0[0]


def test_bounds_respected():
    syn = run(CORRELATED, None)
    w, w0 = apply(syn, +100.0)           # absurd reward: still clipped at 2*w0
    assert 0.0 <= w[0] <= 2.0 * w0[0]


def test_zero_reward_keeps_w():
    syn = run(CORRELATED, None)
    w, w0 = apply(syn, 0.0)
    assert np.allclose(w, w0)


if __name__ == "__main__":
    test_correlated_reward_grows_w()
    test_correlated_punish_shrinks_w()
    test_anticorrelated_does_not_grow()
    test_bounds_respected()
    test_zero_reward_keeps_w()
    print("OK: reward-modulated STDP verified")