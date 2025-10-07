// Dense.h
#ifndef DENSE_H
#define DENSE_H

#include "Matrix.h"
#include "Activation.h"

class Dense
{
public:
    // Type Alias for Activation Function
    using ActivationFunction = Matrix (*)(const Matrix&);

    // Constructor
    Dense(const Matrix& weights, const Matrix& bias,
        ActivationFunction activation);
    // Getters
    Matrix get_weights() const;
    Matrix get_bias() const;
    ActivationFunction get_activation() const;
    // Overload the function call operator to apply the dense layer
    Matrix operator()(const Matrix& input) const;

private:
    Matrix _weights;
    Matrix _bias;
    ActivationFunction _activation;
};

#endif //DENSE_H