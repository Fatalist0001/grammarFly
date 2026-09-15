import numpy as np
from brian2 import SpikeMonitor, ms


class Readout:
    def __init__(self, accept, reject):
        self.accept_spikes = accept if isinstance(accept, SpikeMonitor) else SpikeMonitor(accept)
        self.reject_spikes = reject if isinstance(reject, SpikeMonitor) else SpikeMonitor(reject)

    @staticmethod
    def _count(monitor, t0, t1):
        t = np.asarray(monitor.t / ms)
        if len(t) == 0:
            return 0
        return int(((t >= t0 / ms) & (t <= t1 / ms)).sum())

    def read_output(self, t0, t1):
        accept = self._count(self.accept_spikes, t0, t1)
        reject = self._count(self.reject_spikes, t0, t1)
        return "ACCEPT" if accept > reject else "REJECT"