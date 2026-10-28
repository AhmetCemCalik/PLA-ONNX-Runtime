# PLA with ONNX Runtime

A C++ implementation of the Perceptron Learning Algorithm that generates linearly separable data, trains a perceptron, exports it to ONNX, and performs local or remote inference via REST/gRPC.

## Requirements

- C++17 compiler (`g++`), Python 3, Make, Docker
- Python packages: `numpy`, `onnx`, `onnxruntime`, `grpcio`, `grpcio-tools`
- Generated gRPC Python bindings from the Datasance protocol

## Make Commands

| Command | Description |
|---|---|
| `make input_dim=N` | Train, export ONNX, and test locally (default: 2 features) |
| `make train` | Compile and train the perceptron |
| `make export` | Train and export the ONNX model |
| `make test` | Train, export, and test using ONNX Runtime |
| `make deploy` | Copy the existing ONNX model to `models/pla/` |
| `make generate` | Generate `build/request.json` using the saved input dimension (default: 100 points) |
| `make rest` | Send the JSON request via REST |
| `make grpc` | Send the same JSON request via gRPC |
| `make clean` | Remove generated build files |

Training saves the feature count to `build/input_dim.txt`. Request generation reads it automatically. Configure the number of generated points with `batch_size=N`.

## Workflow

**1. Train and test**

```bash
make input_dim=5
```

**2. Deploy the trained model**

```bash
make deploy
```

**3. Start the Datasance inference engine**

From the project root:

```bash
docker run --rm --read-only --tmpfs /tmp \
  -e AUTH_MODE=disabled \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/models:/models:ro" \
  ghcr.io/datasance/ai-inference-onnx:0.1.0-rc.3-ort1.31.0-cpu
```

Restart the engine after deploying a new model.

**4. Generate inference inputs**

```bash
make generate batch_size=100
```

The input dimension is read from `build/input_dim.txt`.

**5. Run inference**

```bash
make rest
make grpc
```

Both clients use the same `build/request.json` and should produce identical predictions.

Batch size must respect `models/pla/config.pbtxt`.