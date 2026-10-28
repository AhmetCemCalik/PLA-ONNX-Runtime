#include "dataset.hpp"
#include <random>
#include <iostream>

void Dataset::generateLinearData( int input_dim, int n, unsigned int seed ) {
    std::mt19937 rng( seed );

    std::uniform_int_distribution<int> coeffDist( -10, 10 );
    std::uniform_real_distribution<double> pointDist( -10.0, 10.0 );

    std::vector<int> coeff( input_dim + 1 );

    for ( int i = 0; i < input_dim + 1; i++ ) coeff[ i ] = coeffDist( rng );
    std::cout << "Coefficients of generated data:\n";
    for (int c : coeff) std::cout << '\t' << c;
    std::cout << '\n';
    
    points.clear();
    points.reserve( n );

    for ( int i = 0; i < n; ++i ) {
        std::vector<double> xn( input_dim );
        for ( int j = 0; j < input_dim; j++ ) xn[ j ] = pointDist( rng );

        double value = coeff[ 0 ];
        for ( int j = 0; j < input_dim; j++ ) value += coeff[ j + 1 ] * xn[ j ];

        int label = ( value >= 0.0 ) ? 1 : -1;

        points.push_back({xn, label });
    }
}