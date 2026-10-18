CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
PYTHON = python3

input_dim ?= 2

SOURCES = main.cpp dataset.cpp perceptron.cpp

all: test

main: $(SOURCES) perceptron.hpp dataset.hpp
	$(CXX) $(CXXFLAGS) $(SOURCES) -o main

.PHONY: FORCE
FORCE:

weights.txt: main FORCE
	./main $(input_dim)

perceptron.onnx: weights.txt export_onnx.py
	$(PYTHON) export_onnx.py

test: perceptron.onnx test_onnx.py
	$(PYTHON) test_onnx.py $(input_dim)

clean:
	rm -f main weights.txt perceptron.onnx

.PHONY: all test clean