import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope, PoissonGroup, SpikeMonitor, Network

from brain.malecns import MaleCNSGraph
from brain.lif import make_synapse

PA = 0.001 * nA
STIM = 100 * ms
N_IN = 8


def main():
    graph = MaleCNSGraph("data/malecns_eb")
    epg = np.array([i for i, t in enumerate(graph.types) if t == "EPG"])
    er = np.array([i for i, t in enumerate(graph.types) if t.startswith("ER")])
    order = np.argsort(graph.body_ids[epg])
    bump = epg[order][:16]

    print(f"{'config':22s} | stim_sp | epgHz  erHz  | bump/other during stim")
    for w_scale, tau_m, tau_syn, w_in, rate in [
        (0.25 * PA, 10 * ms, 2 * ms, 0.8 * nA, 100 * Hz),
        (0.5 * PA, 10 * ms, 2 * ms, 0.8 * nA, 100 * Hz),
        (1.0 * PA, 10 * ms, 2 * ms, 0.8 * nA, 100 * Hz),
        (0.5 * PA, 20 * ms, 5 * ms, 0.8 * nA, 100 * Hz),
        (1.0 * PA, 20 * ms, 5 * ms, 0.8 * nA, 100 * Hz),
        (1.0 * PA, 30 * ms, 8 * ms, 0.8 * nA, 100 * Hz),
        (0.5 * PA, 10 * ms, 2 * ms, 1.6 * nA, 200 * Hz),
    ]:
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
        t = np.asarray(mon.t / ms)
        i = mon.i
        total = len(t)
        epg_sp = np.isin(i, epg).sum()
        er_sp = np.isin(i, er).sum()
        bu_sp = np.isin(i, bump).sum()
        oth_sp = np.isin(i, np.setdiff1d(epg, bump)).sum()
        print(f"w={w_scale/nA*1000:5.2f}pa t={tau_m/ms:2.0f}/{tau_syn/ms:.0f} "
              f"in={w_in/nA:.1f}nA r={rate/Hz:3.0f}Hz | {total:7d} "
              f"| {epg_sp/0.1/len(epg):6.1f} {er_sp/0.1/len(er):6.1f} "
              f"| bump={bu_sp:5d} other={oth_sp:5d} (bump/all-epg={100*bu_sp/max(epg_sp,1):4.1f}%)")


if __name__ == "__main__":
    main()