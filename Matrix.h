// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H



// You don't have to use the struct. Can help you with MlpNetwork.h
struct matrix_dims {
    int rows, cols;
};

#include <iostream>
#include <string>

class Matrix {
public:
    Matrix(int rows, int cols); // Explicit Constructor
    Matrix(); // Default Constructor (Delegation)
    Matrix(const Matrix& m); // Copy-Constructor
    ~Matrix();
    // Getters
    int get_rows() const;
    int get_cols() const;
    // Methods
    Matrix& transpose();
    Matrix& vectorize();
    void plain_print() const;
    Matrix dot(const Matrix& m) const;
    float norm() const;
    Matrix rref() const;
    int argmax() const;
    float sum() const;
    // Matrix Operators
    Matrix& operator+=(const Matrix& m);
    Matrix operator+(const Matrix& m) const;
    Matrix& operator=(const Matrix& m);
    Matrix operator*(const Matrix& m) const; // matrix multiplication
    Matrix operator*(float scalar) const; // scalar left
    friend Matrix operator*(float scalar, const Matrix& m); // scalar * right
    // Access Operators
    float& operator()(int row, int col);
    const float& operator()(int row, int col) const;
    float& operator[](int index);
    const float& operator[](int index) const;
    // Stream Operators
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
    friend std::istream& operator>>(std::istream& is, Matrix& m);

private:
    int _rows;
    int _cols;
    float* _data;
};

#endif //MATRIX_H