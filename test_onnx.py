import sys
import onnxruntime as ort
import numpy as np

if len( sys.argv ) != 2:
    print(f"Usage: python3 {sys.argv[0]} <input_dim>")
    sys.exit(1)

input_dim = int(sys.argv[1])

if input_dim <= 0:
    print("input_dim must be positive")
    sys.exit(1)

# Load the ONNX model
session = ort.InferenceSession(
    "perceptron.onnx",
    providers=["CPUExecutionProvider"]
)

# Inspect the model's interface
for inp in session.get_inputs():
    print("Input:", inp.name, inp.shape, inp.type)

for out in session.get_outputs():
    print("Output:", out.name, out.shape, out.type)

# Generate 100 random points with "input_dim" features each
rng = np.random.default_rng(42)

x = rng.uniform(
    -10.0, 10.0,
    size=(100, input_dim)
).astype(np.float64)

# Run inference on all 100 points
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: x})

predictions = outputs[0]

# Display results
for i in range(len(x)):
    print(f"Point {i+1}: {x[i]} -> {predictions[i, 0]}")

print("\nTotal points:", len(x))
print("Class +1:", np.sum(predictions == 1))
print("Class -1:", np.sum(predictions == -1))
