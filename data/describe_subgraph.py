import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "malecns_mbRlobes")


def main():
    nodes = pd.read_csv(os.path.join(BASE, "nodes.csv"))
    edges = pd.read_csv(os.path.join(BASE, "edges.csv"))

    pairs = edges.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
    deg = np.concatenate([
        pairs["bodyId_pre"].value_counts().to_numpy(),
        pairs["bodyId_post"].value_counts().to_numpy(),
    ])

    print("nodes:", len(nodes))
    print("edges (per roi rows):", len(edges))
    print("unique pairs:", len(pairs))
    print("mean weight: %.1f min %d max %d" % (
        edges["weight"].mean(), edges["weight"].min(), edges["weight"].max()))
    print("degree: mean %.2f min %d max %d" % (deg.mean(), deg.min(), deg.max()))

    # Giant weakly-connected component
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(nodes["bodyId"])
    g.add_edges_from(zip(pairs["bodyId_pre"], pairs["bodyId_post"]))
    comps = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
    print("weakly connected components:", len(comps),
          "largest:", len(comps[0]) if comps else 0)

    # Cell types
    print("\ntop cell types:")
    print(nodes["type"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()