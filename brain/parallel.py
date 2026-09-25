"""Parallel word scoring with multiprocessing.

Each worker process builds its own copy of the network once (from a spec
snapshot of a PipePipeline) and then scores a batch of words independently.
The simulator layer (Brian2) stays single-threaded; speedup comes from running
many words concurrently on separate CPU cores.

Usage:
    F, y = collect_features_parallel(pipe, samples, seed=1, n_proc=6)
"""

import os

import numpy as np

from brian2 import Hz, ms, nA, start_scope


def _build_pipe(spec):
    from brain.malecns import MaleCNSGraph
    from organs.sensory import SensoryOrgan
    from brain.pipeline import Pipeline

    start_scope()
    np.random.seed(spec["seed"])
    graph = MaleCNSGraph(spec["graph_path"])
    org_a = SensoryOrgan("A", spec["org_a_n"], spec["org_rate"] * Hz)
    org_b = SensoryOrgan("B", spec["org_b_n"], spec["org_rate"] * Hz)
    pipe = Pipeline(
        graph, org_a, org_b,
        np.asarray(spec["in_a"]), np.asarray(spec["in_b"]),
        np.asarray(spec["out_acc"]), np.asarray(spec["out_rej"]),
        spec["w_in"] * nA, spec["w_scale"] * nA,
        plastic=spec["plastic"],
        kc_mbon_plastic=spec["kc_mbon_plastic"],
        kc_mbon_stdp=spec["kc_mbon_stdp"],
        kc_scope=spec["kc_scope"],
        tau_el=spec["tau_el_ms"] * ms,
        a_plus=spec["a_plus"], a_minus=spec["a_minus"],
    )
    if spec["plastic"] and spec.get("w") is not None:
        syn = getattr(pipe, "kc_mbon_syn", None) or pipe.mcns_syn
        syn.w = np.asarray(spec["w"]) * nA
    return pipe


def _worker_init(spec):
    global _PIPE
    _PIPE = _build_pipe(spec)


def _worker_task(word):
    pipe = _PIPE
    pipe.reset_state()
    t0, t1 = pipe.present_word(word)
    return _half_features(pipe, t0, t1)


def _half_features(pipe, t0, t1):
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    if len(t) == 0:
        return np.zeros(8)
    mid = (t0 / ms + t1 / ms) / 2
    sel_early = (t >= t0 / ms) & (t < mid)
    sel_late = (t >= mid) & (t <= t1 / ms)

    def h(sel, idx):
        return np.isin(i[sel], idx).sum()

    n_acc, n_rej = len(pipe.out_acc), len(pipe.out_rej)
    n_a, n_b = len(pipe.in_a), len(pipe.in_b)
    tt = np.unique(t)
    t0s, t1s = t0 / ms, t1 / ms
    early_span = max(tt[(tt >= t0s) & (tt < mid)].max(initial=t0s) - t0s, 0.1) / 1e3
    late_span = max(t1s - tt[(tt >= mid) & (tt <= t1s)].min(initial=t1s), 0.1) / 1e3
    return np.array([
        h(sel_early, pipe.out_acc) / n_acc / early_span,
        h(sel_late, pipe.out_acc) / n_acc / late_span,
        h(sel_early, pipe.out_rej) / n_rej / early_span,
        h(sel_late, pipe.out_rej) / n_rej / late_span,
        h(sel_early, pipe.in_a) / n_a / early_span,
        h(sel_late, pipe.in_a) / n_a / late_span,
        h(sel_early, pipe.in_b) / n_b / early_span,
        h(sel_late, pipe.in_b) / n_b / late_span,
    ])


def collect_features_parallel(pipe, samples, seed=1, n_proc=None, chunksize=4):
    """Score every (word, label) sample in a process pool.

    Returns (F, y) exactly like the sequential collect_features: F is a
    (n_samples, 8) feature matrix, y the labels. Word order is preserved.
    Each worker copies the current plastic weights from `pipe`.
    """
    import multiprocessing as mp

    if n_proc is None:
        n_proc = min(6, max(1, (os.cpu_count() or 2) - 2))
    words = [w for w, _ in samples]
    syn = getattr(pipe, "kc_mbon_syn", None) or pipe.mcns_syn
    spec = {
        "graph_path": pipe.graph.base_dir,
        "seed": int(seed),
        "in_a": np.asarray(pipe.in_a),
        "in_b": np.asarray(pipe.in_b),
        "out_acc": np.asarray(pipe.out_acc),
        "out_rej": np.asarray(pipe.out_rej),
        "w_in": float(pipe.w_in / nA),
        "w_scale": float(pipe.w_scale / nA),
        "org_a_n": len(pipe.organ_a.source),
        "org_b_n": len(pipe.organ_b.source),
        "org_rate": float(pipe.organ_a.rate / Hz),
        "plastic": bool(pipe.plastic),
        "kc_mbon_plastic": bool(pipe.kc_mbon_plastic),
        "kc_mbon_stdp": pipe.kc_mbon_stdp,
        "kc_scope": pipe.kc_scope,
        "tau_el_ms": float(pipe.tau_el / ms),
        "a_plus": float(pipe.a_plus),
        "a_minus": float(pipe.a_minus),
        "w": np.asarray(syn.w / nA) if pipe.plastic else None,
    }
    if n_proc <= 1:
        pipe_local = _build_pipe(spec)
        F = []
        for w in words:
            pipe_local.reset_state()
            t0, t1 = pipe_local.present_word(w)
            F.append(_half_features(pipe_local, t0, t1))
        return np.array(F), np.array([l for _, l in samples])

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_proc, initializer=_worker_init, initargs=(spec,)) as pool:
        F = pool.map(_worker_task, words, chunksize=chunksize)
    return np.array(F), np.array([l for _, l in samples])