
import json
import struct

with open("build/response.headers", "r") as f:
    headers = f.readlines()

header_length = None

for line in headers:
    if line.lower().startswith("inference-header-content-length:"):
        header_length = int(line.split(":", 1)[1].strip())
        break

if header_length is None:
    raise ValueError("Missing Inference-Header-Content-Length")

with open("build/response.bin", "rb") as f:
    body = f.read()

metadata = json.loads(body[:header_length])
tensor_data = body[header_length:]

output = metadata["outputs"][0]

if output["datatype"] != "INT64":
    raise ValueError("Unexpected output datatype")

count = output["parameters"]["binary_data_size"] // 8

predictions = struct.unpack(f"<{count}q", tensor_data[:count * 8])

print("Model:", metadata["model_name"])
print("Output shape:", output["shape"])
print("Number of predictions:", len(predictions))

for i, prediction in enumerate(predictions):
    print(f"Point {i+1}: {prediction}")

print("Class +1:", predictions.count(1))
print("Class -1:", predictions.count(-1))
