import sys

sys.path.insert(0, "D:\\VScode_pr\\grammarFly")

from neuprint_utils import make_client


def main():
    client = make_client()
    all_rois = set(client.all_rois)
    candidates = [
        "MB(R)", "MB(L)", "CAL(R)", "CAL(L)", "PED(R)", "PED(L)",
        "a'L(R)", "alpha'L(R)", "aL(R)", "alphaL(R)", "bL(R)", "b'L(R)",
        "CX", "EB", "FB", "PB", "NO", "AB(R)", "AB(L)",
        "GF(R)", "GF(L)", "GC", "SAT(R)", "SAT(L)", "SLP(R)", "SLP(L)",
        "SEZ", "SNP(R)", "SNP(L)", "GNG",
    ]
    for roi in candidates:
        if roi not in all_rois:
            print(f"{roi:12s}  (not in roi list)")
            continue
        q = "MATCH (n:Neuron) WHERE n.%s RETURN count(n) as c" % _prop(roi)
        frame = client.fetch_custom(q)
        print(f"{roi:12s}  {frame.iloc[0]['c']}")


def _prop(roi):
    return "`%s`" % roi.replace("`", "")


if __name__ == "__main__":
    main()