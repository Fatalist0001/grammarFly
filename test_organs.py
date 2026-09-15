import numpy as np
from brian2 import Hz, ms, second, start_scope

from organs.sensory import Sensor

N_NEURONS = 4
RATE = 100 * Hz
WINDOW = 50 * ms


def count_in_window(monitor, t0, t1):
    t = np.asarray(monitor.t / ms)
    if len(t) == 0:
        return 0
    return int(((t >= t0 / ms) & (t <= t1 / ms)).sum())


def check_symbol(sensor, symbol):
    sensor.present(symbol)
    sym, t0, t1 = sensor.history[-1]
    assert sym == symbol
    active = sensor.organ_a if symbol == "A" else sensor.organ_b
    silent = sensor.organ_b if symbol == "A" else sensor.organ_a
    assert count_in_window(active.spikes, t0, t1) > 0, f"{symbol}: no spikes"
    assert count_in_window(silent.spikes, t0, t1) == 0, f"{symbol}: {silent.symbol} fired"


def test_single_symbols():
    start_scope()
    sensor = Sensor(N_NEURONS, RATE, WINDOW)
    check_symbol(sensor, "A")
    check_symbol(sensor, "B")
    print("A-pattern (ms):", sorted(np.asarray(sensor.organ_a.spikes.t / ms)))
    print("B-pattern (ms):", sorted(np.asarray(sensor.organ_b.spikes.t / ms)))


def test_sequence():
    start_scope()
    sensor = Sensor(N_NEURONS, RATE, WINDOW)
    sensor.present_sequence("AABB")

    assert [h[0] for h in sensor.history] == ["A", "A", "B", "B"]
    for _, t0, t1 in sensor.history:
        assert abs((t1 - t0) - WINDOW) / second < 1e-9

    ta_ms = np.asarray(sensor.organ_a.spikes.t / ms)
    tb_ms = np.asarray(sensor.organ_b.spikes.t / ms)
    assert len(ta_ms) > 0 and len(tb_ms) > 0
    assert ta_ms.max() <= 100.0
    assert tb_ms.min() >= 100.0
    assert tb_ms.max() <= 200.0
    print("A-spikes (ms):", sorted(ta_ms))
    print("B-spikes (ms):", sorted(tb_ms))


if __name__ == "__main__":
    test_single_symbols()
    test_sequence()
    print("OK: organs verified at spike level")