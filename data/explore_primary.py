import sys

sys.path.insert(0, "D:\\VScode_pr\\grammarFly")

from neuprint_utils import make_client


def main():
    client = make_client()
    primaries = set(client.primary_rois)
    print("num primary:", len(primaries))
    for roi in ["MB(R)", "MB(L)", "PED(R)", "a'L(R)", "b'L(R)", "aL(R)", "bL(R)", "EB", "PB", "FB", "NO", "AB(R)", "CX"]:
        print(f"{roi:10s} primary={roi in primaries}")


if __name__ == "__main__":
    main()