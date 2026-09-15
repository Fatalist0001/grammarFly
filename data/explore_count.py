import sys

sys.path.insert(0, "D:\\VScode_pr\\grammarFly")

from neuprint_utils import make_client


def main():
    client = make_client()
    test_rois = ["ANm", "AMMC(L)", "AMMC(R)", "AMMC"]
    for roi in test_rois:
        q = "MATCH (n:Neuron) WHERE n.roiInfo[%s] IS NOT NULL RETURN count(n) as c" % _json_quote(roi)
        r = client.fetch_custom(q)
        print(roi, r.iloc[0]["c"] if len(r) else None)


def _json_quote(s):
    return '"' + s.replace('"', '\\"') + '"'


if __name__ == "__main__":
    main()