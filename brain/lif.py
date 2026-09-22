from brian2 import NeuronGroup, Synapses, amp, Mohm, ms, mV, nA


def make_lif(n, name="neurons", e_l=-70 * mV, v_thresh=-50 * mV,
             v_reset=-65 * mV, tau_m=10 * ms, tau_syn=2 * ms, R=200 * Mohm):
    model = (
        "dv/dt = (e_l - v + R * I) / tau_m : volt\n"
        "dI/dt = -I / tau_syn : amp"
    )
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
        threshold="v > v_thresh",
        reset="v = v_reset; I = 0 * amp",
        namespace=namespace,
        method="euler",
        name=name,
    )
    group.v = e_l
    return group


def make_synapse(pre, post, source_idx=None, target_idx=None, weights=None,
                 w=0.1 * nA, p=None, name="S"):
    syn = Synapses(pre, post, model="w : amp", on_pre="I_post += w", name=name)
    if source_idx is None:
        syn.connect(p=p)
        syn.w = w
    else:
        syn.connect(i=source_idx, j=target_idx)
        syn.w = w if weights is None else weights
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