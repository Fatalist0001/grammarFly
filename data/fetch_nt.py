"""Download predictedNeurotransmitter info for the MB-right-lobe subgraph.

Writes data/malecns_mbRlobes/nt.csv (bodyId, predictedNt, predictedNtConfidence,
consensusNt, status) and prints a distribution summary. Requires network access
to neuprint.janelia.org and a token in .env.
"""
import os

import pandas as pd

from neuprint import Client, fetch_neurons

NODES = "data/malecns_mbRlobes/nodes.csv"
OUT = "data/malecns_mbRlobes/nt.csv"


def load_auth():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")) as f:
        for line in f:
            if line.strip().startswith("AUTH="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("AUTH not found in .env")


if __name__ == "__main__":
    client = Client("https://neuprint.janelia.org",
                    dataset="male-cns:v1.0", token=load_auth())
    nodes = pd.read_csv(NODES)
    ids = sorted(int(b) for b in nodes["bodyId"])
    print(f"fetching predictedNt for {len(ids)} neurons...")
    neurons, _ = fetch_neurons(ids, client=client)
    keep = neurons[["bodyId", "predictedNt", "predictedNtConfidence",
                    "consensusNt", "status", "statusLabel"]].copy()
    keep.to_csv(OUT, index=False)
    print(f"saved -> {OUT} ({len(keep)} rows)")

    nt = neurons["predictedNt"].fillna("(none)")
    print("\npredictedNt distribution (neuron-level):")
    print(nt.value_counts().to_string())
    conf = neurons["predictedNtConfidence"].dropna()
    if len(conf):
        print(f"\nconfidence: n={len(conf)} mean={conf.mean():.3f} min={conf.min():.3f} max={conf.max():.3f}")
    missing = nt.eq("(none)").sum()
    print(f"missing predictedNt: {missing}/{len(neurons)}")