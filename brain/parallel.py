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
        tau_pre=spec["tau_pre_ms"] * ms,
        tau_post=spec["tau_post_ms"] * ms,
        tau_slow=spec["tau_slow_ms"] * ms,
        a_plus_slow=spec["a_plus_slow"],
        a_minus_slow=spec["a_minus_slow"],
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


def _pair_metrics_task(word):
    pipe = _PIPE
    pipe.reset_state()
    t0, t1 = pipe.present_word(word)
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    t0s, t1s = t0 / ms, t1 / ms
    word_sel = (t >= t0s) & (t <= t1s)
    n_spikes = int(word_sel.sum())
    if n_spikes == 0:
        return np.zeros(4), n_spikes
    mid = (t0s + t1s) / 2
    sel_e = (t >= t0s) & (t < mid)
    sel_l = (t >= mid) & (t <= t1s)
    n_acc = len(pipe.out_acc)
    rejE = np.isin(i[sel_e], pipe.out_rej).sum() / n_acc
    rejL = np.isin(i[sel_l], pipe.out_rej).sum() / n_acc
    accE = np.isin(i[sel_e], pipe.out_acc).sum() / n_acc
    accL = np.isin(i[sel_l], pipe.out_acc).sum() / n_acc
    return np.array([
        rejE, rejL, accE, accL,
    ]), n_spikes


def _elig_task(args):
    word, label, reward_mode, channel_reward_only, reward_norm = args
    pipe = _PIPE
    syn = getattr(pipe, "kc_mbon_syn", None) or pipe.mcns_syn
    for var in ("x_fast", "y_fast", "x_slow", "y_slow", "x", "y"):
        if var in syn.variables:
            setattr(syn, var, 0)
    post_idx = np.asarray(syn.j)
    post_acc = np.isin(post_idx, np.asarray(pipe.out_acc))
    post_rej = np.isin(post_idx, np.asarray(pipe.out_rej))
    pipe.reset_state()
    syn.elig = 0
    if reward_mode == "profile":
        n = len(word)
        k = max(1, n // 2)
        organs = {"A": pipe.organ_a, "B": pipe.organ_b}
        t0 = pipe.net.t
        elig_early = np.zeros(len(syn.elig))
        for si, symbol in enumerate(word):
            org = organs[symbol]
            org.source.rates = org.rate
            pipe.net.run(50 * ms)
            org.source.rates = 0 * Hz
            if si + 1 == k:
                elig_early = np.asarray(syn.elig).copy()
        elig_late = np.asarray(syn.elig) - elig_early
        t1 = pipe.net.t
        goal = 1.0 if label else -1.0
        rv_e = np.zeros(len(syn.elig))
        rv_e[post_acc] = -goal
        rv_e[post_rej] = goal
        rv_l = np.zeros(len(syn.elig))
        rv_l[post_acc] = goal
        rv_l[post_rej] = -goal
        contrib = rv_l * elig_late + rv_e * elig_early
    else:
        t0, t1 = pipe.present_word(word)
        goal = 1.0 if label else -1.0
        rv = np.zeros(len(syn.elig))
        rv[post_acc] = goal
        rv[post_rej] = -goal
        if not channel_reward_only:
            rv[~(post_acc | post_rej)] = goal
        elig = np.asarray(syn.elig)
        if reward_norm == "word":
            elig = elig / len(word)
        contrib = rv * elig
    result = pipe.read_output(t0, t1)
    correct = (result == "ACCEPT") == bool(label)
    t = np.asarray(pipe.mon_brain.t / ms)
    n_spikes = int(((t >= t0 / ms) & (t <= t1 / ms)).sum())
    return contrib, correct, n_spikes


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
        "tau_pre_ms": float(pipe.tau_pre / ms),
        "tau_post_ms": float(pipe.tau_post / ms),
        "tau_slow_ms": float(pipe.tau_slow / ms),
        "a_plus_slow": None if pipe.a_plus_slow is None else float(pipe.a_plus_slow),
        "a_minus_slow": None if pipe.a_minus_slow is None else float(pipe.a_minus_slow),
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


def _make_spec(pipe, seed=1):
    syn = getattr(pipe, "kc_mbon_syn", None) or pipe.mcns_syn
    return {
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
        "tau_pre_ms": float(pipe.tau_pre / ms),
        "tau_post_ms": float(pipe.tau_post / ms),
        "tau_slow_ms": float(pipe.tau_slow / ms),
        "a_plus_slow": None if pipe.a_plus_slow is None else float(pipe.a_plus_slow),
        "a_minus_slow": None if pipe.a_minus_slow is None else float(pipe.a_minus_slow),
        "w": np.asarray(syn.w / nA) if pipe.plastic else None,
    }


def collect_metrics_parallel(pipe, words, seed=1, n_proc=None, chunksize=4):
    """Return (M, n_spikes): M is (n_words, 4) [rejE, rejL, accE, accL]."""
    import multiprocessing as mp

    if n_proc is None:
        n_proc = min(6, max(1, (os.cpu_count() or 2) - 2))
    spec = _make_spec(pipe, seed=seed)
    if n_proc <= 1:
        p1 = _build_pipe(spec)
        M, NS = [], []
        for w in words:
            p1.reset_state()
            t0, t1 = p1.present_word(w)
            m, ns = _pair_metrics_task_ts(p1, t0, t1)
            M.append(m)
            NS.append(ns)
        return np.array(M), np.array(NS)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_proc, initializer=_worker_init, initargs=(spec,)) as pool:
        out = pool.map(_pair_metrics_task, list(words), chunksize=chunksize)
    M = np.array([o[0] for o in out])
    NS = np.array([o[1] for o in out])
    return M, NS


def collect_elig_parallel(pipe, samples, seed=1, n_proc=None, chunksize=4,
                          reward_mode="class", channel_reward_only=False,
                          reward_norm=None):
    """Run trainer-style reward accumulation for every sample in workers.

    Returns (contrib, corrects, n_spikes): contrib is (n, n_syn) matrix of
    rv*elig, corrects boolean per word, n_spikes total per word.
    Workers build the network once from a snapshot including current weights.
    """
    import multiprocessing as mp

    if n_proc is None:
        n_proc = min(6, max(1, (os.cpu_count() or 2) - 2))
    words = [(w, l, reward_mode, channel_reward_only, reward_norm) for w, l in samples]
    spec = _make_spec(pipe, seed=seed)
    if n_proc <= 1:
        p1 = _build_pipe(spec)
        out = []
        for w, l, rm, cro, rn in words:
            out.append(_elig_task((w, l, rm, cro, rn)))
        return (np.array([o[0] for o in out]), np.array([o[1] for o in out]),
                np.array([o[2] for o in out]))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_proc, initializer=_worker_init, initargs=(spec,)) as pool:
        out = pool.map(_elig_task, words, chunksize=chunksize)
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]),
            np.array([o[2] for o in out]))


class ParallelTrainer:
    """Persistent process pool hosting one network build per worker.

    Weights are updated in-place via `set_weights` before each epoch, so the
    network is built only once per worker -- no spawn overhead per epoch.
    """

    def __init__(self, pipe, seed=1, n_proc=None):
        import multiprocessing as mp

        self._seed = seed
        if n_proc is None:
            n_proc = min(6, max(1, (os.cpu_count() or 2) - 2))
        self.n_proc = n_proc
        self.spec = _make_spec(pipe, seed=seed)
        self.ctx = mp.get_context("spawn")
        self.pool = self.ctx.Pool(processes=n_proc, initializer=_worker_init,
                                  initargs=(self.spec,))
        self._nsyn = int(np.asarray(self.spec["w"]).size)

    def set_weights(self, w):
        self.spec["w"] = np.asarray(w, dtype=float)
        args = [(i, self.spec["w"]) for i in range(self.n_proc)]
        self.pool.map(_set_weights_task, args, chunksize=1)

    def fit_epoch(self, samples, reward_mode="class",
                  channel_reward_only=False, reward_norm=None):
        words = [(w, l, reward_mode, channel_reward_only, reward_norm)
                 for w, l in samples]
        out = self.pool.map(_elig_task, words, chunksize=max(1, len(words) // self.n_proc))
        contrib = np.array([o[0] for o in out])
        corrects = np.array([o[1] for o in out])
        n_spikes = np.array([o[2] for o in out])
        return contrib, corrects, n_spikes

    def metrics(self, words):
        out = self.pool.map(_pair_metrics_task, list(words), chunksize=4)
        M = np.array([o[0] for o in out])
        NS = np.array([o[1] for o in out])
        return M, NS

    def close(self):
        self.pool.close()
        self.pool.join()


def _set_weights_task(args):
    _, w = args
    pipe = _PIPE
    syn = getattr(pipe, "kc_mbon_syn", None) or pipe.mcns_syn
    syn.w = np.asarray(w) * nA


def _pair_metrics_task_ts(pipe, t0, t1):
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    if len(t) == 0:
        return np.zeros(4)
    mid = (t0 / ms + t1 / ms) / 2
    sel_e = (t >= t0 / ms) & (t < mid)
    sel_l = (t >= mid) & (t <= t1 / ms)
    n_acc = len(pipe.out_acc)
    rejE = np.isin(i[sel_e], pipe.out_rej).sum() / n_acc
    rejL = np.isin(i[sel_l], pipe.out_rej).sum() / n_acc
    accE = np.isin(i[sel_e], pipe.out_acc).sum() / n_acc
    accL = np.isin(i[sel_l], pipe.out_acc).sum() / n_acc
    return np.array([rejE, rejL, accE, accL])