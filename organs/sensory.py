from brian2 import PoissonGroup, SpikeMonitor, Network, Hz, ms


class SensoryOrgan:
    def __init__(self, symbol, n_neurons=4, rate=50 * Hz):
        self.symbol = symbol
        self.rate = rate
        self.source = PoissonGroup(n_neurons, rates=0 * Hz)
        self.spikes = SpikeMonitor(self.source)


class Sensor:
    def __init__(self, n_neurons=4, rate=50 * Hz, window=50 * ms):
        self.window = window
        self.organ_a = SensoryOrgan("A", n_neurons, rate)
        self.organ_b = SensoryOrgan("B", n_neurons, rate)
        self.network = Network(
            self.organ_a.source,
            self.organ_a.spikes,
            self.organ_b.source,
            self.organ_b.spikes,
        )
        self.history = []

    def present(self, symbol):
        t0 = self.network.t
        if symbol == "A":
            organ = self.organ_a
        elif symbol == "B":
            organ = self.organ_b
        else:
            raise ValueError(f"unknown symbol: {symbol!r}")
        organ.source.rates = organ.rate
        self.network.run(self.window)
        organ.source.rates = 0 * Hz
        self.history.append((symbol, t0, self.network.t))

    def present_sequence(self, sequence):
        for symbol in sequence:
            self.present(symbol)