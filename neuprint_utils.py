import os

from neuprint import Client


def load_auth():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    with open(env_path) as f:
        for line in f:
            if line.strip().startswith("AUTH="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("AUTH not found in .env")


def make_client(dataset="male-cns:v1.0"):
    return Client(
        "https://neuprint.janelia.org",
        dataset=dataset,
        token=load_auth(),
    )