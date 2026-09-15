import sys

sys.path.insert(0, "D:\\VScode_pr\\grammarFly")

from neuprint_utils import make_client


def main():
    client = make_client()
    q = "MATCH (n:Neuron)-[e:Contains]->(r) RETURN labels(r) as lbl, keys(r) as k LIMIT 3"
    df = client.fetch_custom(q)
    print(df.to_string())


if __name__ == "__main__":
    main()