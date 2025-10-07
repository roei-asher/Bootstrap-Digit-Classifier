#include "Dense.h"
#include "Matrix.h"

//************************** Constructor *************************************
Dense::Dense(const Matrix& weights, const Matrix& bias,
    ActivationFunction activation)
    : _weights(weights), _bias(bias), _activation(activation) {}
//****************************************************************************


//**************************** Getters ***************************************
Matrix Dense::get_weights() const {return _weights;}

Matrix Dense::get_bias() const {return _bias;}

Dense::ActivationFunction Dense::get_activation() const {return _activation;}
//****************************************************************************


//*********************** Dense Layer Operator *******************************
Matrix Dense::operator()(const Matrix& input) const
{
    const Matrix z = (_weights * input) + _bias;
    return _activation(z);
}
//****************************************************************************