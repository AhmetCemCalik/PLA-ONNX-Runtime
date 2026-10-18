# PLA with ONNX Runtime

A simple C++ implementation of the Perceptron Learning Algorithm (PLA) that generates linearly separable data, trains a perceptron, exports the learned model to ONNX, and runs inference using ONNX Runtime.

## Requirements

- C++17 compiler (`g++`)
- Python 3
- Python packages: `numpy`, `onnx`, `onnxruntime`
- Make

## Run

```bash
make input_dim=5
```

Generates a dataset, trains a perceptron with 5 input features, exports the model to ONNX, and tests inference on 100 random points.

The default input dimension is 2:

```bash
make
```

To remove generated files:

```bash
make clean
```