
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
PYTHON = .venv/bin/python

input_dim ?= 2
batch_size ?= 100
request ?= build/request.json

SRC = src
SCRIPTS = scripts
BUILD = build
MODEL_DIR = models/pla

SOURCES = $(SRC)/main.cpp $(SRC)/dataset.cpp $(SRC)/perceptron.cpp
HEADERS = $(SRC)/dataset.hpp $(SRC)/perceptron.hpp

MODEL = $(BUILD)/perceptron.onnx

# Default: train, export, and test locally
all: test

# Directories
$(BUILD):
	mkdir -p $(BUILD)

$(MODEL_DIR):
	mkdir -p $(MODEL_DIR)

# Compile C++
$(BUILD)/main: $(SOURCES) $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

# Train PLA and save input dimension
$(BUILD)/weights.txt: $(BUILD)/main FORCE
	./$(BUILD)/main $(input_dim)
	echo $(input_dim) > $(BUILD)/input_dim.txt

train: $(BUILD)/weights.txt

# Export to ONNX
$(MODEL): $(BUILD)/weights.txt $(SCRIPTS)/export_onnx.py
	$(PYTHON) $(SCRIPTS)/export_onnx.py

export: $(MODEL)

# Test using local ONNX Runtime
test: $(MODEL)
	$(PYTHON) $(SCRIPTS)/test_onnx.py $(input_dim)

# Deploy model to serving directory
deploy: | $(MODEL_DIR)
	cp $(MODEL) $(MODEL_DIR)/perceptron.onnx

# Generate inference request
generate: | $(BUILD)
	$(PYTHON) $(SCRIPTS)/generate_requests.py \
		--batch-size $(batch_size) \
		--input-dim $$(cat $(BUILD)/input_dim.txt) \
		--output $(request)

# REST inference
rest:
	$(PYTHON) $(SCRIPTS)/REST_client.py $(request)

# gRPC inference
grpc:
	$(PYTHON) $(SCRIPTS)/gRPC_client.py $(request)

# Clean generated files
clean:
	rm -rf $(BUILD)

FORCE:

.PHONY: all train export test deploy generate rest grpc clean FORCE
