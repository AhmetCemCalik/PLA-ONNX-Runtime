# Sends JSON requests to Datasance inference engine over HTTP

import argparse
import json
import urllib.request
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("file", help="JSON inference request")
parser.add_argument(
    "--url",
    default="http://127.0.0.1:8000/v2/models/pla/infer"
)
args = parser.parse_args()

# Read JSON request
with open(args.file) as f:
    payload = json.load(f)

# Construct HTTP request
request = urllib.request.Request(
    args.url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

# Send request
with urllib.request.urlopen(request, timeout=30) as response:
    body = response.read()
    header_length = response.headers.get(
        "Inference-Header-Content-Length"
    )

# Separate metadata and binary output
if header_length is not None:
    length = int(header_length)
    metadata = json.loads(body[:length])
    binary = body[length:]
else:
    metadata = json.loads(body)
    binary = None

# Decode output tensor
for output in metadata["outputs"]:
    if output["name"] != "Y":
        continue

    if output["datatype"] != "INT64":
        raise ValueError("Expected INT64 output")

    shape = output["shape"]

    if binary is not None:
        size = output["parameters"]["binary_data_size"]

        if len(binary) != size:
            raise ValueError("Unexpected binary output size")

        predictions = np.frombuffer(
            binary, dtype="<i8"
        ).reshape(shape)
    else:
        predictions = np.array(
            output["data"], dtype=np.int64
        ).reshape(shape)

    print("Model:", metadata["model_name"])
    print("Output shape:", predictions.shape)
    print("Predictions:")
    print(predictions.flatten())

    print("Class +1:", np.sum(predictions == 1))
    print("Class -1:", np.sum(predictions == -1))
