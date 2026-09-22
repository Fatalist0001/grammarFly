import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope, PoissonGroup, SpikeMonitor, Network

from brain.malecns import MaleCNSGraph
from brain.lif import make_synapse

PA = 0.001 * nA
STIM = 100 * ms
SILENCE = 500 * ms
BINS = [(0, 50), (50, 100), (100, 250), (250, 500)]
N_IN = 8
COMBOS = []
for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
    for tm, ts in [(10, 2), (15, 3), (20, 5)]:
        COMBOS.append((w * PA, tm * ms, ts * ms, 0.8 * nA, 100 * Hz))


def main():
    graph = MaleCNSGraph("data/malecns_eb")
    epg = np.array([i for i, t in enumerate(graph.types) if t == "EPG"])
    order = np.argsort(graph.body_ids[epg])
    bump = epg[order][:16]
    other = epg[order][16:]

    print(f"{'config':20s} | after-stim bins (bump/total EPG, bump% of EPG spikes)")
    for w_scale, tau_m, tau_syn, w_in, rate in COMBOS:
        start_scope()
        np.random.seed(0)
        brain, syn = graph.build(w_scale=w_scale, plastic=False,
                                 tau_m=tau_m, tau_syn=tau_syn)
        org = PoissonGroup(N_IN, rates=rate, name="dir_org")
        s_in = make_synapse(org, brain, np.arange(N_IN),
                            bump[:N_IN], np.ones(N_IN) * w_in, name="dir_input")
        mon = SpikeMonitor(brain)
        net = Network(brain, syn, org, s_in, mon)
        net.run(STIM)
        org.rates = 0 * Hz
        t_start = net.t / ms
        net.run(SILENCE)
        t = np.asarray(mon.t / ms)
        i = mon.i
        label = f"w={w_scale/nA*1000:4.1f} t={tau_m/ms:2.0f}/{tau_syn/ms:.0f}"
        cells = []
        for a, b in BINS:
            w0, w1 = t_start + a, t_start + b
            sel = (t >= w0) & (t <= w1)
            nb = np.isin(i[sel], bump).sum()
            no = np.isin(i[sel], other).sum()
            tot = nb + no
            q = f"{nb:4d}/{tot:5d}({100*nb/max(tot,1):3.0f}%)"
            cells.append(q)
        print(f"{label:20s} | " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()