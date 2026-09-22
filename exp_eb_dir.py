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
K = 4
SIL_START = 100 * ms  # skip initial transient after input-off


def run_dir(graph, targets, w_scale, tau_m, tau_syn, w_in, rate):
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
    ts0 = net.t / ms
    net.run(SILENCE)
    t = np.asarray(mon.t / ms)
    i = mon.i
    return brain, t, i, ts0


def epg_vec(t, i, epg, ts0):
    sel = (t >= ts0 + SIL_START / ms) & (t <= ts0 + (SIL_START + SILENCE) / ms)
    return np.array([np.isin(i[sel], e).sum() for e in epg], dtype=float)


def cosmatch(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def main():
    graph = MaleCNSGraph("data/malecns_eb")
    epg = np.array([i for i, t in enumerate(graph.types) if t == "EPG"])
    order = np.argsort(graph.body_ids[epg])
    epg_sorted = epg[order]
    dirA = epg_sorted[:12]
    dirB = epg_sorted[23:35]
    print(f"dirA(0-11)={dirA} dirB(23-34)={dirB}")

    print(f"\n{'config':18s} {'intraA':>6s} {'intraB':>6s} {'inter':>6s} "
          f"{'inter/intraA':>11s} {'sepA2B?':>7s} {'locA%':>6s} {'locB%':>6s}")
    for w_scale, tm, ts in [(1.0 * PA, 20 * ms, 5 * ms),
                            (1.2 * PA, 20 * ms, 5 * ms),
                            (1.5 * PA, 20 * ms, 5 * ms),
                            (0.9 * PA, 20 * ms, 5 * ms),
                            (1.0 * PA, 15 * ms, 3 * ms)]:
        for dname, targets in [("A", dirA), ("B", dirB)]:
            globals()[f"prof_{dname}"] = []
            globals()[f"loc_{dname}"] = []
            base = epg[:16]
            for k in range(K):
                brain, t, i, ts0 = run_dir(graph, targets, w_scale, tm, ts,
                                           0.8 * nA, 100 * Hz)
                v = epg_vec(t, i, epg, ts0)
                globals()[f"prof_{dname}"].append(v / (v.sum() + 1e-12))
                sel = (t >= ts0 + SIL_START / ms) & (t <= ts0 + (SIL_START + SILENCE) / ms)
                bu = np.isin(i[sel], targets).sum()
                ot = np.isin(i[sel], np.setdiff1d(epg, targets)).sum()
                globals()[f"loc_{dname}"].append(100 * bu / (bu + ot + 1e-9))
        PA_ = np.array(prof_A); PB_ = np.array(prof_B)
        intraA = np.mean([1 - cosmatch(PA_[i], PA_[j]) for i in range(K) for j in range(i + 1, K)]) if K > 1 else 0
        intraB = np.mean([1 - cosmatch(PB_[i], PB_[j]) for i in range(K) for j in range(i + 1, K)]) if K > 1 else 0
        inter = np.mean([1 - cosmatch(PA_[i], PB_[j]) for i in range(K) for j in range(K)])
        sep = inter / max(intraA + intraB, 1e-9)
        print(f"w={w_scale/nA*1000:.1f} t={tm/ms:.0f}/{ts/ms:.0f} "
              f"{intraA:6.3f} {intraB:6.3f} {inter:6.3f} {sep:11.1f} "
              f"{'YES' if sep > 2 else 'no' :>7s} "
              f"{np.mean(loc_A):5.1f}% {np.mean(loc_B):5.1f}%", flush=True)
        # inter should exceed intra if directions stay distinct


if __name__ == "__main__":
    main()