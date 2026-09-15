import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuprint_utils import make_client


def main():
    client = make_client()
    print("ROIs:", len(client.all_rois))

    q = """
    MATCH (n:Neuron)-[:Contains]->(r:ROI)
    RETURN r.roi as roi, count(distinct n) as cells
    ORDER BY cells DESC
    """
    df = client.fetch_custom(q)
    print(df.to_string())


if __name__ == "__main__":
    main()