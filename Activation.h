// activation.h
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"
#include <string>

namespace activation
{
    Matrix relu(const Matrix& input);
    Matrix softmax(const Matrix& input);
}

#endif //ACTIVATION_H