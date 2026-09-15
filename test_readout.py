import numpy as np
from brian2 import Hz, ms, start_scope

from organs.readout import Readout
from brian2 import PoissonGroup, Network


def run_case(accept_rate, reject_rate, n_neurons=4):
    start_scope()
    accept = PoissonGroup(n_neurons, rates=0 * Hz)
    reject = PoissonGroup(n_neurons, rates=0 * Hz)
    readout = Readout(accept, reject)
    net = Network(accept, reject, readout.accept_spikes, readout.reject_spikes)
    accept.rates = accept_rate
    reject.rates = reject_rate
    window = 50 * ms
    net.run(window)
    return readout.read_output(0 * ms, window)


def test_accept():
    assert run_case(300 * Hz, 5 * Hz) == "ACCEPT"


def test_reject():
    assert run_case(5 * Hz, 300 * Hz) == "REJECT"


def test_no_activity():
    assert run_case(0 * Hz, 0 * Hz) == "REJECT"


def test_counts_are_windowed():
    start_scope()
    accept = PoissonGroup(4, rates=0 * Hz)
    reject = PoissonGroup(4, rates=100 * Hz)
    readout = Readout(accept, reject)
    net = Network(accept, reject, readout.accept_spikes, readout.reject_spikes)
    net.run(50 * ms)
    reject.rates = 0 * Hz
    accept.rates = 300 * Hz
    net.run(50 * ms)
    assert readout.read_output(0 * ms, 50 * ms) == "REJECT"
    assert readout.read_output(50 * ms, 100 * ms) == "ACCEPT"
    assert readout.read_output(0 * ms, 100 * ms) == "ACCEPT"


if __name__ == "__main__":
    test_accept()
    test_reject()
    test_no_activity()
    test_counts_are_windowed()
    print("OK: readout verified")