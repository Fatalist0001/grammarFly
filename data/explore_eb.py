import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_adjacencies
from neuprint_utils import make_client

ROIS = ["EB"]
# actually also PB maybe; let's first look at EB alone and EB+PB combos


def main():
    client = make_client()

    print("=== EB: neurons by type ===")
    criteria = NC(rois=["EB"], roi_req="any", label="Neuron", client=client)
    neurons, _ = fetch_neurons(criteria, client=client)
    print("total EB neurons:", len(neurons))
    vc = neurons["type"].value_counts()
    print(vc.head(40).to_string())
    print("... unique types:", len(vc))

    print()
    print("=== EB: connections within EB (intra) ===")
    neurons, conns = fetch_adjacencies(
        criteria, criteria,
        rois=["EB"], min_roi_weight=1, min_total_weight=1,
        include_nonprimary=False, omit_rois=False,
        batch_size=200, threads=4,
        properties=["type"],
        client=client,
    )
    conns = conns[conns["roi"].isin(["EB"])].reset_index(drop=True)
    print("intra-EB edges:", len(conns))
    caj = conns.merge(neurons[["bodyId", "type"]].rename(columns={"type": "pre_type"}),
                      left_on="bodyId_pre", right_on="bodyId", how="left")
    caj = caj.merge(neurons[["bodyId", "type"]].rename(columns={"type": "post_type"}),
                    left_on="bodyId_post", right_on="bodyId", how="left")
    # exclude self (R-neurons often connect to themselves in ring)
    caj["pre_t"] = caj["pre_type"].astype(str).str.extract(r"^([A-Za-z0-9]+)")[0]
    caj["post_t"] = caj["post_type"].astype(str).str.extract(r"^([A-Za-z0-9]+)")[0]
    cross = caj.groupby(["pre_t", "post_t"]).agg(n=("weight", "size"), s=("weight", "sum"))
    print(cross.sort_values("n", ascending=False).head(40).to_string())

    print()
    print("=== EB neuron types sample ===")
    for t in vc.head(12).index:
        sub = neurons[neurons["type"] == t][["bodyId", "instance"]]
        print(t, "->", len(sub), "instances:", sub.head(6)["instance"].tolist())


if __name__ == "__main__":
    main()