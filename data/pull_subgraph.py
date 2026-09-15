import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_adjacencies

from neuprint_utils import make_client

ROIS = ["aL(R)", "a'L(R)", "bL(R)", "b'L(R)"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "malecns_mbRlobes")


def main():
    client = make_client()
    os.makedirs(OUT_DIR, exist_ok=True)

    criteria = NC(rois=ROIS, roi_req="any", label="Neuron", client=client)
    neurons, _ = fetch_neurons(criteria, client=client)
    print("neurons in MB right lobes:", len(neurons))

    neurons, conns = fetch_adjacencies(
        criteria,
        criteria,
        rois=ROIS,
        min_roi_weight=1,
        min_total_weight=1,
        include_nonprimary=False,
        omit_rois=False,
        batch_size=200,
        threads=4,
        properties=["type", "instance"],
        client=client,
    )
    conns = conns.copy()
    conns = conns[conns["roi"].isin(ROIS)].reset_index(drop=True)
    print("intra-lobe connections (per roi):", len(conns))

    nodes = neurons[["bodyId", "type", "instance"]].copy()
    nodes = nodes.sort_values("bodyId").reset_index(drop=True)
    edges = conns[["bodyId_pre", "bodyId_post", "roi", "weight"]].copy()
    edges = edges.sort_values(["bodyId_pre", "bodyId_post"]).reset_index(drop=True)

    nodes.to_csv(os.path.join(OUT_DIR, "nodes.csv"), index=False)
    edges.to_csv(os.path.join(OUT_DIR, "edges.csv"), index=False)

    print("saved:", os.path.basename(OUT_DIR))
    print("  nodes:", len(nodes))
    print("  edges:", len(edges))
    print("  edges per set of pairs:", edges.groupby(["bodyId_pre", "bodyId_post"]).size().shape[0])

    nodes_with_roi = set(edges["bodyId_pre"]) | set(edges["bodyId_post"])
    isolated = set(nodes["bodyId"]) - nodes_with_roi
    print("  isolated neurons:", len(isolated))


if __name__ == "__main__":
    main()