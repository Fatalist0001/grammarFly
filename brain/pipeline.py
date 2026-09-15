import numpy as np
from brian2 import Network, SpikeMonitor, Hz, amp, ms, mV, nA

from brain.lif import make_synapse

WINDOW = 50 * ms


class Pipeline:
    def __init__(self, graph, organ_a, organ_b, in_a, in_b, out_acc, out_rej,
                 w_in, w_scale, plastic=False, kc_mbon_plastic=False, name="mcns"):
        self.graph = graph
        self.organ_a = organ_a
        self.organ_b = organ_b
        self.in_a = np.asarray(in_a)
        self.in_b = np.asarray(in_b)
        self.out_acc = np.asarray(out_acc)
        self.out_rej = np.asarray(out_rej)
        self.w_in = w_in
        self.w_scale = w_scale
        self.plastic = plastic
        self.kc_mbon_plastic = kc_mbon_plastic

        build_result = graph.build(name=name, w_scale=w_scale,
                                   plastic=plastic, kc_mbon_plastic=kc_mbon_plastic)
        if kc_mbon_plastic:
            self.brain, self.mcns_syn_static, self.kc_mbon_syn = build_result
            self.mcns_syn = self.mcns_syn_static  # backward compat
        else:
            self.brain, self.mcns_syn = build_result
            self.kc_mbon_syn = None

        self.s_in_a = self._organ_synapse(organ_a, self.in_a)
        self.s_in_b = self._organ_synapse(organ_b, self.in_b)

        self.mon_sensors_a = SpikeMonitor(organ_a.source, name="mon_sens_a")
        self.mon_sensors_b = SpikeMonitor(organ_b.source, name="mon_sens_b")
        self.mon_brain = SpikeMonitor(self.brain, name="mon_brain")

        net_objs = [
            organ_a.source, organ_b.source,
            organ_a.spikes, organ_b.spikes,
            self.brain,
        ]
        if kc_mbon_plastic:
            net_objs.extend([self.mcns_syn_static, self.kc_mbon_syn])
        else:
            net_objs.append(self.mcns_syn)
        net_objs.extend([self.s_in_a, self.s_in_b,
                         self.mon_sensors_a, self.mon_sensors_b, self.mon_brain])

        self.net = Network(*net_objs)

    def _organ_synapse(self, organ, targets):
        n_org = len(organ.source)
        i = np.tile(np.arange(n_org), len(targets))
        j = np.repeat(targets, n_org)
        weights = np.ones(len(i)) * self.w_in
        return make_synapse(organ.source, self.brain, i, j, weights,
                            name=f"{organ.symbol}_input")

    def reset_weights(self, w_scale):
        self.w_scale = w_scale
        if self.kc_mbon_plastic:
            self.mcns_syn_static.w = self.graph.pairs["weight"].to_numpy() * w_scale
            kc_mbon = self.graph.kc_mbon_pairs()
            self.kc_mbon_syn.w = kc_mbon["weight"].to_numpy() * w_scale
        else:
            self.mcns_syn.w = self.graph.pairs["weight"].to_numpy() * w_scale

    def reset_state(self, e_l=-70 * mV):
        self.brain.v = e_l
        self.brain.I = 0 * amp

    def present_word(self, word):
        t0 = self.net.t
        organs = {"A": self.organ_a, "B": self.organ_b}
        segments = []
        for sym in word:
            if segments and segments[-1][0] == sym:
                segments[-1] = (sym, segments[-1][1] + 1)
            else:
                segments.append((sym, 1))
        for sym, count in segments:
            org = organs[sym]
            org.source.rates = org.rate
            self.net.run(count * WINDOW)
            org.source.rates = 0 * Hz
        return t0, self.net.t

    def read_output(self, t0, t1):
        t = np.asarray(self.mon_brain.t / ms)
        i = self.mon_brain.i
        if len(t) == 0:
            return "REJECT"
        sel = (t >= t0 / ms) & (t <= t1 / ms)
        acc = np.isin(i[sel], self.out_acc).sum() / len(self.out_acc)
        rej = np.isin(i[sel], self.out_rej).sum() / len(self.out_rej)
        return "ACCEPT" if acc > rej else "REJECT"

    def rate(self, indices, t0, t1):
        if t1 <= t0:
            return 0.0
        t = np.asarray(self.mon_brain.t / ms)
        i = self.mon_brain.i
        idx = np.asarray(indices)
        if len(t) == 0:
            return 0.0
        sel = (t >= t0 / ms) & (t <= t1 / ms) & np.isin(i, idx)
        return sel.sum() / ((t1 - t0) / ms * 1e-3) / len(idx)