import numpy as np
from brian2 import NeuronGroup, Synapses, amp, Mohm, ms, mV, nA, volt


IZ_PRESETS = {
    "rs": {"a": 0.02, "b": 0.2, "c": -65 * mV, "d": 8.0 * mV},   # regular spiking
    "ib": {"a": 0.02, "b": 0.2, "c": -55 * mV, "d": 4.0 * mV},   # intrinsically bursting
    "ch": {"a": 0.02, "b": 0.2, "c": -50 * mV, "d": 2.0 * mV},   # chattering (bistable-ish)
}


def make_izhikevich(n, name="neurons", preset="rs", a=0.02, b=0.2,
                    c=-65 * mV, d=8.0 * mV, v_peak=30 * mV, tau_syn=2 * ms,
                    R=200 * Mohm, V_BASE=0.0 * mV):
    if preset is not None:
        p = IZ_PRESETS[preset]
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    model = (
        "dv/dt = ((0.04 / mV) * v * v + 5.0 * v + 140.0 * mV - u + R * I) "
        "/ (1.0 * ms) : volt\n"
        "du/dt = a * (b * v - u) / (1.0 * ms) : volt\n"
        "dI/dt = -I / tau_syn : amp"
    )
    threshold = "v >= v_peak"
    reset = "v = c; u += d"
    namespace = {
        "a": a, "b": b, "c": c, "d": d,
        "v_peak": v_peak, "tau_syn": tau_syn, "R": R,
    }
    group = NeuronGroup(
        n, model, threshold=threshold, reset=reset, namespace=namespace,
        method="euler", name=name,
    )
    group.v = c + V_BASE
    group.u = b * (c + V_BASE)
    group.I = 0 * amp
    return group


def make_lif(n, name="neurons", e_l=-70 * mV, v_thresh=-50 * mV,
             v_reset=-65 * mV, tau_m=10 * ms, tau_syn=2 * ms, R=200 * Mohm,
             tau_ahp=None, w_ahp=0 * mV):
    if tau_ahp is not None:
        model = (
            "dv/dt = (e_l - v + R * I - u) / tau_m : volt\n"
            "dI/dt = -I / tau_syn : amp\n"
            "du/dt = -u / tau_ahp : volt"
        )
        threshold = "v > v_thresh"
        reset = "v = v_reset; I = 0 * amp; u += w_ahp"
        namespace = {
            "e_l": e_l,
            "v_thresh": v_thresh,
            "v_reset": v_reset,
            "tau_m": tau_m,
            "tau_syn": tau_syn,
            "R": R,
            "tau_ahp": tau_ahp,
            "w_ahp": w_ahp,
        }
    else:
        model = (
            "dv/dt = (e_l - v + R * I) / tau_m : volt\n"
            "dI/dt = -I / tau_syn : amp"
        )
        threshold = "v > v_thresh"
        reset = "v = v_reset; I = 0 * amp"
        namespace = {
            "e_l": e_l,
            "v_thresh": v_thresh,
            "v_reset": v_reset,
            "tau_m": tau_m,
            "tau_syn": tau_syn,
            "R": R,
        }
    group = NeuronGroup(
        n,
        model,
        threshold=threshold,
        reset=reset,
        namespace=namespace,
        method="euler",
        name=name,
    )
    group.v = e_l
    return group


def make_synapse(pre, post, source_idx=None, target_idx=None, weights=None,
                 w=0.1 * nA, p=None, inh=None, name="S"):
    if inh is not None:
        model = "w : amp\nsgn : 1"
        on_pre = "I_post += w * sgn"
    else:
        model = "w : amp"
        on_pre = "I_post += w"
    syn = Synapses(pre, post, model=model, on_pre=on_pre, name=name)
    if source_idx is None:
        syn.connect(p=p)
        syn.w = w
    else:
        syn.connect(i=source_idx, j=target_idx)
        syn.w = w if weights is None else weights
    if inh is not None:
        syn.sgn = np.where(np.asarray(inh), -1, 1)
    return syn


def make_stdp_synapse(pre, post, source_idx, target_idx, weights,
                      tau_el=20 * ms, a_plus=0.01, a_minus=0.01, name="S"):
    syn = Synapses(
        pre,
        post,
        model=(
            "w : amp\n"
            "d elig / dt = -elig / tau_el : 1 (clock-driven)"
        ),
        on_pre=(
            "I_post += w\n"
            "elig += a_plus"
        ),
        on_post="elig -= a_minus",
        namespace={"tau_el": tau_el, "a_plus": a_plus, "a_minus": a_minus},
        method="euler",
        name=name,
    )
    syn.connect(i=source_idx, j=target_idx)
    syn.w = weights
    syn.elig = 0
    return syn


def make_trace_stdp_synapse(pre, post, source_idx, target_idx, weights,
                            tau_el=20 * ms, tau_pre=20 * ms, tau_post=20 * ms,
                            a_plus=0.01, a_minus=0.01, name="S"):
    syn = Synapses(
        pre,
        post,
        model=(
            "w : amp\n"
            "dx/dt = -x / tau_pre : 1 (clock-driven)\n"
            "dy/dt = -y / tau_post : 1 (clock-driven)\n"
            "d elig / dt = -elig / tau_el : 1 (clock-driven)"
        ),
        on_pre=(
            "I_post += w\n"
            "x += 1\n"
            "elig += a_minus * y"
        ),
        on_post=(
            "y += 1\n"
            "elig += a_plus * x"
        ),
        namespace={
            "tau_el": tau_el,
            "tau_pre": tau_pre,
            "tau_post": tau_post,
            "a_plus": a_plus,
            "a_minus": a_minus,
        },
        method="euler",
        name=name,
    )
    syn.connect(i=source_idx, j=target_idx)
    syn.w = weights
    syn.elig = 0
    return syn