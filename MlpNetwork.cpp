#include "MlpNetwork.h"


//************************** Constructor *************************************
MlpNetwork::MlpNetwork(const Matrix weights[MLP_SIZE],
                       const Matrix biases[MLP_SIZE]):
_layers{
    Dense(weights[0], biases[0], activation::relu),
    Dense(weights[1], biases[1], activation::relu),
    Dense(weights[2], biases[2], activation::relu),
    Dense(weights[3], biases[3], activation::softmax)
        }{}
//****************************************************************************


//********************** Apply the MLP network *******************************
digit MlpNetwork::operator()(const Matrix& input) const
{
    // Copy Input (so as not to alter it)
    Matrix output = input;
    output.vectorize();

    // Complete all forward passes of MLP network
    for (int i = 0; i < MLP_SIZE; ++i) {
        output = _layers[i](output);
    }
    // Find the digit with the highest probability
    digit result;
    result.value = output.argmax();
    result.probability = output[result.value];

    return result;
}
//****************************************************************************