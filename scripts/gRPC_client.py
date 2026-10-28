# Sends requests to the Datasance inference engine using gRPC

import argparse
import json
import grpc
import numpy as np

import open_inference_grpc_pb2 as pb2
import open_inference_grpc_pb2_grpc as pb2_grpc

parser = argparse.ArgumentParser()
parser.add_argument("file", help="JSON inference request")
parser.add_argument(
    "--server",
    default="127.0.0.1:8001"
)
parser.add_argument("--model", default="pla")

args = parser.parse_args()

# Read the same JSON used by REST
with open(args.file) as f:
    payload = json.load(f)

# Construct gRPC inference request
request = pb2.ModelInferRequest(model_name=args.model)

for inp in payload["inputs"]:
    if inp["datatype"] != "FP64":
        raise ValueError("Expected FP64 input")

    shape = inp["shape"]
    data = np.asarray(inp["data"], dtype="<f8")

    if data.size != int(np.prod(shape)):
        raise ValueError("Input data does not match shape")

    # Add tensor metadata
    tensor = request.inputs.add()
    tensor.name = inp["name"]
    tensor.datatype = inp["datatype"]
    tensor.shape.extend(shape)

    # Encode flattened tensor as raw bytes
    request.raw_input_contents.append(data.tobytes())

# Connect to inference engine
with grpc.insecure_channel(args.server) as channel:
    stub = pb2_grpc.GRPCInferenceServiceStub(channel)

    # Check model readiness
    ready = stub.ModelReady(
        pb2.ModelReadyRequest(name=args.model),
        timeout=10
    )

    if not ready.ready:
        raise RuntimeError("Model is not ready")

    # Execute inference
    response = stub.ModelInfer(request, timeout=30)

# Decode output tensors
for i, output in enumerate(response.outputs):
    if output.name != "Y":
        continue

    if output.datatype != "INT64":
        raise ValueError("Expected INT64 output")

    shape = list(output.shape)

    if len(response.raw_output_contents) > i:
        # Binary tensor response
        predictions = np.frombuffer(
            response.raw_output_contents[i],
            dtype="<i8"
        ).reshape(shape)
    else:
        # Typed protobuf tensor response
        predictions = np.asarray(
            output.contents.int64_contents,
            dtype=np.int64
        ).reshape(shape)

    print("Model:", response.model_name)
    print("Output shape:", predictions.shape)
    print("Predictions:")
    print(predictions.flatten())

    print("Class +1:", np.sum(predictions == 1))
    print("Class -1:", np.sum(predictions == -1))
