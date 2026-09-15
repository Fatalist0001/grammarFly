import sys

sys.path.insert(0, "D:\\VScode_pr\\grammarFly")

from neuprint_utils import make_client


def main():
    client = make_client()

    queries = [
        ("rel-types", "MATCH (n:Neuron)-[e]->(m) RETURN type(e) as t, labels(m) as lbl, count(*) as cnt ORDER BY cnt DESC LIMIT 40"),
        ("roi-nodes", "MATCH (n:Neuron)-[e:Contains]->(r) RETURN r.roi as roi, count(distinct n) as cells ORDER BY cells DESC LIMIT 80"),
    ]
    for name, q in queries:
        print("=== " + name + " ===")
        try:
            df = client.fetch_custom(q)
            print(df.to_string())
        except Exception as exc:
            print("ERR", type(exc).__name__, str(exc)[-300:])


if __name__ == "__main__":
    main()