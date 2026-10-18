#include <random>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "dataset.hpp"
#include "perceptron.hpp"

int main( int argc, char* argv[] ) 
{
    if ( argc != 2 ) {
        std::cerr << "Usage: " << argv[ 0 ] << " <input_dim>\n";
        return 1;
    }

    int input_dim = std::stoi(argv[1]);

    if ( input_dim <= 0 ) {
        std::cerr << "Input dim must be positive\n";
        return 1;
    }
    std::random_device rd;
    Dataset dataset;

    dataset.generateLinearData(input_dim, 1000, rd());

    Perceptron pla(input_dim, 1000);

    pla.train(dataset.getPoints());
    pla.printWeights();
    std::ofstream out("weights.txt");
    out << std::setprecision(17);

    for (double w : pla.getWeights()) {
        out << w << '\n';
    }

    return 0;
}