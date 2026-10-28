#pragma once
#include <vector>

struct DataPoint {
    std::vector<double> features;
    int label;
};

class Dataset {
private:
    std::vector<DataPoint> points;

public:
    void generateLinearData(int input_dim, int n, unsigned int seed);

    const std::vector<DataPoint>& getPoints() const {
        return points;
    }
};