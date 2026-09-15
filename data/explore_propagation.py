import pandas as pd
import numpy as np

BASE = "data/malecns_mbRlobes"
nodes = pd.read_csv(f"{BASE}/nodes.csv")
edges = pd.read_csv(f"{BASE}/edges.csv")

types = nodes.set_index("bodyId")["type"]
edges["type_pre"] = edges["bodyId_pre"].map(types)
edges["type_post"] = edges["bodyId_post"].map(types)

pairs = edges.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum().reset_index()
pairs["type_pre"] = pairs["bodyId_pre"].map(types)
pairs["type_post"] = pairs["bodyId_post"].map(types)

kc = pairs["type_pre"].str.startswith("KC")
kc_post = pairs["bodyId_post"].isin(nodes[nodes["type"].str.startswith("KC")]["bodyId"])
mbon_post = pairs["bodyId_post"].isin(nodes[nodes["type"].str.startswith("MBON")]["bodyId"])

print("pairs total:", len(pairs))
print("KC -> KC:", int((kc & kc_post).sum()))
print("KC -> MBON:", int((kc & mbon_post).sum()))
print("KC -> PAM:", int((kc & pairs["type_post"].str.startswith("PAM")).sum()))

out_deg = pairs[kc].groupby("bodyId_pre")["weight"].sum()
print("\nKC total out-weight quantiles per KC:", np.quantile(out_deg, [0.5, 0.9, 0.99]).tolist())

kc_mbon = pairs[kc & mbon_post].groupby("bodyId_post")["weight"].sum()
print("MBON total in-weight quantiles:", np.quantile(kc_mbon, [0.5, 0.9, 0.99]).tolist())
print("num MBON receiving from KCs:", len(kc_mbon))

up = pairs["bodyId_pre"].isin(nodes[nodes["type"].str.startswith("MBON")]["bodyId"])
print("MBON -> (any):", int(up.sum()))
print("weight quantiles all pairs:", np.quantile(pairs["weight"], [0.5, 0.9, 0.99, 1.0]).tolist())