#include <iostream>
#include "perceptron.hpp"

int Perceptron::getSign(double val) 
{
    return (val >= 0) ? 1 : -1;
}

Perceptron::Perceptron(int input_dim, int max_epochs)
    : weights(input_dim + 1, 0.0),
      max_epochs(max_epochs) {}

void Perceptron::train(const std::vector<DataPoint>& dataPoints) 
{

    for (int epoch = 0; epoch < max_epochs; epoch++) {

        bool converged = true;

        for (const auto& point : dataPoints) {

            double pointVal = weights[0];

            for (size_t i = 0; i < point.features.size(); i++) {
                pointVal += point.features[i] * weights[i + 1];
            }

            int prediction = getSign(pointVal);

            if (prediction != point.label) {
                converged = false;

                weights[0] += point.label;

                for (size_t i = 0; i < point.features.size(); i++) {
                    weights[i + 1] += point.features[i] * point.label;
                }
            }
        }

        if (converged) {
            std::cout << "Converged successfully at epoch: "
                      << epoch << '\n';
            return;
        }
    }
}

int Perceptron::predict(const DataPoint& dataPoint) const 
{
    double pointVal = weights[0];

    for (size_t i = 0; i < dataPoint.features.size(); i++) {
        pointVal += dataPoint.features[i] * weights[i + 1];
    }

    return getSign(pointVal);
}

void Perceptron::printWeights()
{
    std::cout << "Bias: " << weights[ 0 ] << '\n';
    for ( size_t i = 1; i < weights.size(); i++ ) {
        std::cout << "w_" << i << ": " << weights[ i ] << '\n';
    }
}