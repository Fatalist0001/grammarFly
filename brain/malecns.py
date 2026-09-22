import os

import numpy as np
import pandas as pd
from brian2 import ms, nA

from brain.lif import make_lif, make_synapse


class MaleCNSGraph:
    def __init__(self, base_dir, nodes_name="nodes.csv", edges_name="edges.csv"):
        self.base_dir = base_dir
        self.nodes = pd.read_csv(f"{base_dir}/{nodes_name}")
        self.edges = pd.read_csv(f"{base_dir}/{edges_name}")
        self.body_ids = self.nodes["bodyId"].to_numpy()
        self.types = self.nodes["type"].fillna("").to_numpy()
        self.instances = self.nodes["instance"].fillna("").to_numpy()
        self.idx = {int(b): i for i, b in enumerate(self.body_ids)}

        self.nt = {}
        nt_path = f"{base_dir}/nt.csv"
        if os.path.exists(nt_path):
            nt_df = pd.read_csv(nt_path)
            for _, row in nt_df.iterrows():
                nt = str(row.get("consensusNt", ""))
                if not nt or nt == "nan":
                    nt = str(row.get("predictedNt", ""))
                if not nt or nt == "nan":
                    nt = "unclear"
                self.nt[int(row["bodyId"])] = nt

        self.pairs = (
            self.edges
            .groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
            .sum()
            .sort_values(["bodyId_pre", "bodyId_post"])
            .reset_index(drop=True)
        )

    @property
    def n_neurons(self):
        return len(self.nodes)

    @property
    def n_pairs(self):
        return len(self.pairs)

    def node_rois(self):
        rois = {}
        for bid in self.body_ids:
            rois[int(bid)] = sorted({
                r
                for r in self.edges.query("bodyId_pre == @bid or bodyId_post == @bid")["roi"]
            })
        return rois

    def kc_indices(self, n, seed=0):
        kc = [i for i, t in enumerate(self.types) if t.startswith(("KCab", "KCg", "KCa"))]
        rng = np.random.default_rng(seed)
        rng.shuffle(kc)
        return np.array(kc[:n]), np.array(kc[n:2 * n])

    def inh_mask(self, inhibitory_nt=("gaba",)):
        """Boolean array over self.pairs rows: True if the PRE neuron releases an
        inhibitory neurotransmitter (default: gaba). Falls back to APL-only
        masking when no NT data is present."""
        pre_body = self.pairs["bodyId_pre"].to_numpy()
        if self.nt:
            return np.array([
                self.nt.get(int(b), "unclear") in inhibitory_nt
                for b in pre_body
            ], dtype=bool)
        return np.array([
            isinstance(t, str) and t.startswith("APL")
            for t in self.types[[self.idx[int(b)] for b in pre_body]]
        ])

    def mbon_indices(self):
        split_csv = f"{self.base_dir}/readout_split.csv"
        if os.path.exists(split_csv):
            split = pd.read_csv(split_csv)
            acc = [self.idx[int(b)] for b in split.query("channel == 'ACCEPT'")["bodyId"]]
            rej = [self.idx[int(b)] for b in split.query("channel == 'REJECT'")["bodyId"]]
            return np.array(acc), np.array(rej)
        mbons = sorted(
            (i for i, t in enumerate(self.types) if t.startswith("MBON")),
            key=lambda i: self.types[i],
        )
        mid = len(mbons) // 2
        return np.array(mbons[:mid]), np.array(mbons[mid:])

    def kc_mbon_pairs(self):
        kc = {i for i, t in enumerate(self.types) if t.startswith(("KCab", "KCg", "KCa"))}
        mbon = {i for i, t in enumerate(self.types) if t.startswith("MBON")}
        pre_body = self.pairs["bodyId_pre"].to_numpy()
        post_body = self.pairs["bodyId_post"].to_numpy()
        pre_idx = np.array([self.idx[b] for b in pre_body])
        post_idx = np.array([self.idx[b] for b in post_body])
        mask = np.isin(pre_idx, list(kc)) & np.isin(post_idx, list(mbon))
        return self.pairs[mask].reset_index(drop=True)

    def kc_out_pairs(self):
        kc = {i for i, t in enumerate(self.types) if t.startswith(("KCab", "KCg", "KCa"))}
        pre_body = self.pairs["bodyId_pre"].to_numpy()
        pre_idx = np.array([self.idx[b] for b in pre_body])
        mask = np.isin(pre_idx, list(kc))
        return self.pairs[mask].reset_index(drop=True)

    def fb_mask(self, to_where=("KC", "MBON")):
        """Boolean array over self.pairs rows: True if the PRE neuron is an MBON
        and the POST neuron is in to_where (default: feedback legs MBON->KC and
        MBON->MBON)."""
        kc = {i for i, t in enumerate(self.types) if t.startswith(("KCab", "KCg", "KCa"))}
        mbon = {i for i, t in enumerate(self.types) if t.startswith("MBON")}
        pre_idx = np.array([self.idx[int(b)] for b in self.pairs["bodyId_pre"]])
        post_idx = np.array([self.idx[int(b)] for b in self.pairs["bodyId_post"]])
        is_pre_mbon = np.isin(pre_idx, list(mbon))
        post_sets = []
        for w in to_where:
            if w == "KC":
                post_sets.append(np.isin(post_idx, list(kc)))
            elif w == "MBON":
                post_sets.append(np.isin(post_idx, list(mbon)))
        is_post = np.logical_or.reduce(post_sets)
        return is_pre_mbon & is_post

    def kc_kk_mask(self):
        """Boolean over self.pairs rows: True for KC->KC recurrent pairs."""
        kc = {i for i, t in enumerate(self.types) if t.startswith(("KCab", "KCg", "KCa"))}
        pre_idx = np.array([self.idx[int(b)] for b in self.pairs["bodyId_pre"]])
        post_idx = np.array([self.idx[int(b)] for b in self.pairs["bodyId_post"]])
        return np.isin(pre_idx, list(kc)) & np.isin(post_idx, list(kc))

    def build(self, name="mcns", w_scale=0.001 * nA, plastic=False,
              kc_mbon_plastic=False, kc_mbon_stdp="single",
              kc_scope="mbon",
              tau_el=20 * ms, a_plus=0.01, a_minus=0.01, inh_scale=1.0,
              fb_boost=1.0, kc_kk_scale=1.0,
              **lif_kwargs):
        pre = np.array([self.idx[int(b)] for b in self.pairs["bodyId_pre"]])
        post = np.array([self.idx[int(b)] for b in self.pairs["bodyId_post"]])
        inh = self.inh_mask()
        fb = self.fb_mask()
        kk = self.kc_kk_mask()
        weight_scale = np.ones(len(self.pairs))
        if kc_kk_scale != 1.0:
            weight_scale[kk] *= kc_kk_scale
        if fb_boost != 1.0:
            weight_scale[fb] *= fb_boost
        if inh.any():
            weight_scale[inh] *= inh_scale
        weights = self.pairs["weight"].to_numpy() * w_scale * weight_scale
        neurons = make_lif(self.n_neurons, name=name, **lif_kwargs)

        if kc_mbon_plastic:
            # Build static synapses for all non-KC-output
            if kc_scope == "kc_out":
                kc_pairs = self.kc_out_pairs()
            else:
                kc_pairs = self.kc_mbon_pairs()
            kc_pre = np.array([self.idx[int(b)] for b in kc_pairs["bodyId_pre"]])
            kc_post = np.array([self.idx[int(b)] for b in kc_pairs["bodyId_post"]])
            kc_weights = kc_pairs["weight"].to_numpy() * w_scale

            kc_mask = np.zeros(len(self.pairs), dtype=bool)
            kc_key = set(zip(kc_pairs["bodyId_pre"], kc_pairs["bodyId_post"]))
            for r, (b0, b1) in enumerate(zip(self.pairs["bodyId_pre"],
                                              self.pairs["bodyId_post"])):
                kc_mask[r] = (int(b0), int(b1)) in kc_key
            non_mask = ~kc_mask
            non_pre = pre[non_mask]
            non_post = post[non_mask]
            non_weights = (
                self.pairs["weight"].to_numpy()[non_mask]
                * w_scale * weight_scale[non_mask]
            )
            non_inh = inh[non_mask]

            from brain.lif import make_stdp_synapse, make_trace_stdp_synapse
            syn_static = make_synapse(neurons, neurons, non_pre, non_post, non_weights,
                                      inh=np.asarray(non_inh, dtype=bool),
                                      name=f"{name}_syn_static")
            if kc_mbon_stdp == "trace":
                syn_plastic = make_trace_stdp_synapse(
                    neurons, neurons, kc_pre, kc_post, kc_weights,
                    tau_el=tau_el, a_plus=a_plus, a_minus=a_minus,
                    name=f"{name}_kc_mbon_plastic")
            else:
                syn_plastic = make_stdp_synapse(
                    neurons, neurons, kc_pre, kc_post, kc_weights,
                    tau_el=tau_el, a_plus=a_plus, a_minus=a_minus,
                    name=f"{name}_kc_mbon_plastic")
            return neurons, syn_static, syn_plastic
        else:
            if plastic:
                from brain.lif import make_stdp_synapse
                syn = make_stdp_synapse(neurons, neurons, pre, post, weights,
                                        name=f"{name}_syn")
            else:
                syn = make_synapse(neurons, neurons, pre, post, weights,
                                   inh=inh, name=f"{name}_syn")
            return neurons, syn