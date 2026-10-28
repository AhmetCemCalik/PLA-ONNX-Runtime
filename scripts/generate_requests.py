# Generates random data points with the specified parameters.
# Places the generated data into a KServe V2 compatible JSON
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--input-dim", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="build/request.json")

args = parser.parse_args()

if args.batch_size <= 0 or args.input_dim <= 0:
    parser.error("Batch size and input dimension must be positive")

# Generate random points
rng = np.random.default_rng(args.seed)

points = rng.uniform(
    -10.0, 10.0,
    size=(args.batch_size, args.input_dim)
)

# KServe V2 inference request
request = {
    "inputs": [
        {
            "name": "X",
            "shape": [args.batch_size, args.input_dim],
            "datatype": "FP64",
            "data": points.flatten().tolist()
        }
    ]
}

path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)

with path.open("w") as f:
    json.dump(request, f, indent=2)

print(f"Generated {args.batch_size} points with "
      f"{args.input_dim} features.")
print(f"Saved to {path}")
