import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

cluster_url = os.getenv("QDRANT_URL").replace("6334", "6333")
api_key = os.getenv("QDRANT_API_KEY")

headers = {"api-key": api_key}

response = requests.get(
    f"{cluster_url}/telemetry",
    headers=headers,
    params={"details_level": 10},
)
response.raise_for_status()

data = response.json()["result"]

# Extract resource usage
memory = data.get("memory", {})
collections = data.get("collections", {})

print("=== Memory ===")
print(json.dumps(memory, indent=2))

print("\n=== Collections (disk/vector stats) ===")
print(json.dumps(collections, indent=2))
