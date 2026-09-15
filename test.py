import os

from neuprint import Client, fetch_neurons, fetch_adjacencies

def load_auth():
    with open(os.path.join(os.path.dirname(__file__), ".env")) as f:
        for line in f:
            if line.strip().startswith("AUTH="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("AUTH not found in .env")

TOKEN = load_auth()

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=TOKEN,
)

# Получаем информацию о нейронах типа DNge104
neurons, synapses = fetch_neurons("DNge104", client=client)

print("=== NEURONS ===")
print(neurons)

print("\n=== SYNAPSES ===")
print(synapses)

# Получаем связи этих нейронов с другими
outgoing, neuron_info = fetch_adjacencies(
    "DNge104",
    client=client,
)

print("\n=== OUTGOING CONNECTIONS ===")
print(outgoing.head(20))

print(f"\nNeurons: {len(neurons)}")
print(f"Connections: {len(outgoing)}")