#pragma once

#include <vector>
#include "dataset.hpp"

class Perceptron {
private:
    std::vector<double> weights;
    int max_epochs;

    static int getSign(double val);

public:
    Perceptron(int input_dim, int max_epochs);

    void train(const std::vector<DataPoint>& dataPoints);

    int predict(const DataPoint& dataPoint) const;

    const std::vector<double> getWeights() const { return weights; };
    void printWeights();
};