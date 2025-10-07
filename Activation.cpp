#include "Activation.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

//*************************** Errors *****************************************
#define INVALID_FUNC std::invalid_argument("Invalid activation function.")
//****************************************************************************

//************************* Activation Functions *************************
namespace activation
{
    Matrix relu(const Matrix& input)
    {
        Matrix result(input.get_rows(), input.get_cols());
        for (int i = 0; i < input.get_rows() * input.get_cols(); ++i)
            {
            result[i] = std::max(0.0F, input[i]); // Apply ReLU
            }
        return result;
    }

    Matrix softmax(const Matrix& input)
    {
        Matrix result(input.get_rows(), input.get_cols());
        float sum_exp = 0.0;

        // Compute the exponential values and their sum
        for (int i = 0; i < input.get_rows() * input.get_cols(); ++i)
        {
            result[i] = std::exp(input[i]);
            sum_exp += result[i];
        }

        // Normalize by dividing by the sum of exponentials
        if (sum_exp != 0.0)  // Add a check for zero
        {
            for (int i = 0; i < input.get_rows() * input.get_cols(); ++i)
            {
                result[i] /= sum_exp;
            }
        }
        else
        {
            // Uniform Distribution
            float uniform_prob = 1.0F / (input.get_rows() * input.get_cols());
            for (int i = 0; i < input.get_rows() * input.get_cols(); ++i)
            {
                result[i] = uniform_prob;
            }
        }
        return result;
    }
}
//************************************************************************