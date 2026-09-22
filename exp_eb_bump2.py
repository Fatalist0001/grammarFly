import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope, PoissonGroup, SpikeMonitor, Network

from brain.malecns import MaleCNSGraph
from brain.lif import make_synapse

PA = 0.001 * nA
STIM = 100 * ms
SILENCE = 500 * ms
N_IN = 8


def run_drive(graph, targets, w_scale, tau_m, tau_syn, w_in, rate):
    start_scope()
    np.random.seed(0)
    brain, syn = graph.build(w_scale=w_scale, plastic=False,
                             tau_m=tau_m, tau_syn=tau_syn)
    org = PoissonGroup(N_IN, rates=rate, name="dir_org")
    s_in = make_synapse(org, brain, np.arange(N_IN), targets[:N_IN],
                        np.ones(N_IN) * w_in, name="dir_input")
    mon = SpikeMonitor(brain)
    net = Network(brain, syn, org, s_in, mon)
    net.run(STIM)
    org.rates = 0 * Hz
    t_start = net.t / ms
    net.run(SILENCE)
    t = np.asarray(mon.t / ms)
    i = mon.i
    return brain, t, i, t_start


def epg_profile(t, i, epg, t_start):
    out = {}
    for a, b in [(0, 100), (100, 300), (300, 500)]:
        sel = (t >= t_start + a) & (t <= t_start + b)
        counts = np.array([np.isin(i[sel], e).sum() for e in epg])
        tot = counts.sum()
        top8 = np.sort(counts)[::-1][:8].sum()
        out[(a, b)] = (counts, tot, top8)
    return out


def main():
    graph = MaleCNSGraph("data/malecns_eb")
    epg = np.array([i for i, t in enumerate(graph.types) if t == "EPG"])
    order = np.argsort(graph.body_ids[epg])
    bump = epg[order][:16]
    er_all = np.array([i for i, t in enumerate(graph.types) if t.startswith("ER")])
    er_order = np.argsort(graph.body_ids[er_all])
    er_targets = er_all[er_order][:16]

    print("=== drive EPG directly ===")
    for w_scale, tm, ts in [(1.0 * PA, 20 * ms, 5 * ms), (1.5 * PA, 20 * ms, 5 * ms)]:
        brain, t, i, ts0 = run_drive(graph, bump, w_scale, tm, ts, 0.8 * nA, 100 * Hz)
        for (a, b), (counts, tot, top8) in epg_profile(t, i, epg, ts0).items():
            base = epg[:16]
            bu = counts[np.isin(epg, base)].sum()
            print(f"w={w_scale/nA*1000:.1f} t={tm/ms:.0f}/{ts/ms:.0f} [{a:>3d},{b:>3d})ms "
                  f"epg_sp={tot:4d} bump%={100*bu/max(tot,1):3.0f} top8%={100*top8/max(tot,1):3.0f} "
                  f"peak%={100*counts.max()/max(tot,1):.0f}")

    print("\n=== drive ER (ring input) instead ===")
    for w_scale, tm, ts in [(1.0 * PA, 20 * ms, 5 * ms), (1.5 * PA, 20 * ms, 5 * ms),
                            (2.0 * PA, 20 * ms, 5 * ms), (2.0 * PA, 10 * ms, 2 * ms)]:
        brain, t, i, ts0 = run_drive(graph, er_targets, w_scale, tm, ts, 0.8 * nA, 100 * Hz)
        for (a, b), (counts, tot, top8) in epg_profile(t, i, epg, ts0).items():
            print(f"w={w_scale/nA*1000:.1f} t={tm/ms:.0f}/{ts/ms:.0f} [{a:>3d},{b:>3d})ms "
                  f"epg_sp={tot:4d} top8%={100*top8/max(tot,1):3.0f} "
                  f"peak%={100*counts.max()/max(tot,1):.0f}")


if __name__ == "__main__":
    main()