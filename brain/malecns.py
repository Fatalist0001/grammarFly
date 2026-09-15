import numpy as np
import pandas as pd
from brian2 import nA

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

    def mbon_indices(self):
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

    def build(self, name="mcns", w_scale=0.001 * nA, plastic=False, kc_mbon_plastic=False, **lif_kwargs):
        pre = np.array([self.idx[int(b)] for b in self.pairs["bodyId_pre"]])
        post = np.array([self.idx[int(b)] for b in self.pairs["bodyId_post"]])
        weights = self.pairs["weight"].to_numpy() * w_scale
        neurons = make_lif(self.n_neurons, name=name, **lif_kwargs)

        if kc_mbon_plastic:
            # Build static synapses for all non-KC->MBON
            kc_mbon = self.kc_mbon_pairs()
            kc_mbon_pre = np.array([self.idx[int(b)] for b in kc_mbon["bodyId_pre"]])
            kc_mbon_post = np.array([self.idx[int(b)] for b in kc_mbon["bodyId_post"]])
            kc_mbon_weights = kc_mbon["weight"].to_numpy() * w_scale

            non_kc_mbon = self.pairs[~self.pairs.index.isin(kc_mbon.index)]
            non_pre = np.array([self.idx[int(b)] for b in non_kc_mbon["bodyId_pre"]])
            non_post = np.array([self.idx[int(b)] for b in non_kc_mbon["bodyId_post"]])
            non_weights = non_kc_mbon["weight"].to_numpy() * w_scale

            from brain.lif import make_stdp_synapse
            syn_static = make_synapse(neurons, neurons, non_pre, non_post, non_weights,
                                      name=f"{name}_syn_static")
            syn_plastic = make_stdp_synapse(neurons, neurons, kc_mbon_pre, kc_mbon_post, kc_mbon_weights,
                                            name=f"{name}_kc_mbon_plastic")
            return neurons, syn_static, syn_plastic
        else:
            if plastic:
                from brain.lif import make_stdp_synapse
                syn = make_stdp_synapse(neurons, neurons, pre, post, weights,
                                        name=f"{name}_syn")
            else:
                syn = make_synapse(neurons, neurons, pre, post, weights,
                                   name=f"{name}_syn")
            return neurons, syn