#include "Matrix.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

#define PRETTY_PRINT_THRESHOLD 0.1F

//*************************** Errors *****************************************
#define INDEX_ERROR std::out_of_range("ERROR: Matrix index out of range.")
#define DIM_ERROR1 std::invalid_argument("ERROR: Dimensions must be positive.")
#define DIM_ERROR2 std::invalid_argument("ERROR: Incompatible Dimensions.")
#define DATA_ERROR std::ios_base::failure("ERROR: Insufficient data.")
#define DATA_READ_ERROR std::ios_base::failure("Failed to read matrix data.")
//****************************************************************************


//*************************** Constructors ***********************************
Matrix::Matrix(int rows, int cols) :
_rows(rows), _cols(cols), _data(new float[rows * cols]())
{
    if (rows <= 0 || cols <= 0){throw DIM_ERROR1;}
}

Matrix::Matrix() : Matrix(1, 1) {}

Matrix::Matrix(const Matrix& m) :
_rows(m._rows), _cols(m._cols), _data(new float[m._rows * m._cols])
{
    std::copy(m._data, m._data + _rows * _cols, _data);
}
//****************************************************************************


//**************************** Destructor ************************************
Matrix::~Matrix() {delete[] _data;}
//****************************************************************************


//*************************** Getters ****************************************
int Matrix::get_rows() const {return _rows;}

int Matrix::get_cols() const {return _cols;}
//****************************************************************************


//**************************** Methods ***************************************
Matrix& Matrix::transpose()
{
    float* new_data = new float[_rows * _cols];
    // A_ij => A_ji
    for (int i = 0; i < _rows; ++i)
        {
        for (int j = 0; j < _cols; ++j)
            {
            new_data[j * _rows + i] = _data[i * _cols + j];
            }
        }
    std::swap(_rows, _cols);
    delete[] _data;
    _data = new_data;
    return *this;
}

Matrix& Matrix::vectorize()
{
    _rows *= _cols;
    _cols = 1;
    return *this;
}

void Matrix::plain_print() const
{
    for (int i = 0; i < _rows; ++i)
        {
        for (int j = 0; j < _cols; ++j)
            {
            std::cout << _data[i * _cols + j] << " ";
            }
        std::cout << std::endl;
        }
}

Matrix Matrix::dot(const Matrix& m) const
{
    if (_rows != m._rows || _cols != m._cols){throw DIM_ERROR2;}
    Matrix result(_rows, _cols);
    for (int i = 0; i < _rows * _cols; ++i)
        {
        result._data[i] = _data[i] * m._data[i];
        }
    return result;
}

float Matrix::norm() const
{
    float sum_sq = 0.0;
    for (int i = 0; i < _rows * _cols; ++i){sum_sq += _data[i] * _data[i];}
    return std::sqrt(sum_sq);
}

Matrix Matrix::rref() const
{
    // Copy original matrix
    Matrix result(*this);
    int lead = 0; // Lead column
    for (int r = 0; r < _rows; ++r) // Iterate on rows
        {
        if (lead >= _cols) {return result;} // Ensure end
        int i = r;
        while (result(i, lead) == 0) // Traverse to non reduced row
            {
            ++i;
            if (i == _rows)
                {
                i = r;
                ++lead;
                if (lead == _cols) {return result;} // Ensure end
                }
            }
        for (int j = 0; j < _cols; ++j) // Swap rows
            {
            std::swap(result(i, j), result(r, j));
            }
        float lead_val = result(r, lead);
        for (int j = 0; j < _cols; ++j) {result(r, j) /= lead_val;} // Reduce
        for (int i = 0; i < _rows; ++i)  // Adjust other rows
            {
            if (i != r)
                {
                lead_val = result(i, lead);
                for (int j = 0; j < _cols; ++j)
                    {
                    result(i, j) -= lead_val * result(r, j);
                    }
                }
            }
        ++lead; // Repeat algorithm until end
        }
    return result;
}

int Matrix::argmax() const
{
    int max_idx = 0;
    for (int i = 1; i < _rows * _cols; ++i)
        {
        if (_data[i] > _data[max_idx]){max_idx = i;}
        }
    return max_idx;
}

float Matrix::sum() const
{
    float total = 0.0;
    for (int i = 0; i < _rows * _cols; ++i) {total += _data[i];}
    return total;
}
//****************************************************************************


//*************************** Matrix Operators *******************************
Matrix& Matrix::operator+=(const Matrix& m)
{
    if (_rows != m._rows || _cols != m._cols){throw DIM_ERROR2;}
    for (int i = 0; i < _rows * _cols; ++i){_data[i] += m._data[i];}
    return *this;
}

Matrix Matrix::operator+(const Matrix& m) const
{
    Matrix result(*this);
    result += m;
    return result;
}

Matrix& Matrix::operator=(const Matrix& m)
{
    // Reflexive Case
    if (this == &m) {return *this;}
    // Non-Reflexive Case
    delete[] _data;
    _rows = m._rows;
    _cols = m._cols;
    _data = new float[_rows * _cols];
    std::copy(m._data, m._data + _rows * _cols, _data);
    return *this;
}

Matrix Matrix::operator*(const Matrix& m) const
{
    if (_cols != m._rows){throw DIM_ERROR2;}
    Matrix result(_rows, m._cols);
    for (int i = 0; i < _rows; ++i)
        {
        for (int j = 0; j < m._cols; ++j)
            {
            for (int k = 0; k < _cols; ++k)
                {
                result(i, j) += (*this)(i, k) * m(k, j);
                }
            }
        }
    return result;
}

Matrix Matrix::operator*(float scalar) const
{
    Matrix result(_rows, _cols);
    for (int i = 0; i < _rows * _cols; ++i)
        {
        result._data[i] = _data[i] * scalar;
        }
    return result;
}

Matrix operator*(float scalar, const Matrix& m){return m * scalar;}
//****************************************************************************


//************************* Access Operators *********************************
float& Matrix::operator()(int row, int col)
{
    if (row >= _rows || col >= _cols || row < 0 || col < 0){throw INDEX_ERROR;}
    return _data[row * _cols + col];
}

const float& Matrix::operator()(int row, int col) const
{
    if (row >= _rows || col >= _cols || row < 0 || col < 0)
        {
        throw INDEX_ERROR;
        }
    return _data[row * _cols + col];
}

float& Matrix::operator[](int index)
{
    if (index >= _rows * _cols || index < 0) {throw INDEX_ERROR;}
    return _data[index];
}

const float& Matrix::operator[](int index) const
{
    if (index >= _rows * _cols || index < 0) {throw INDEX_ERROR;}
    return _data[index];
}
//****************************************************************************


//***************************** Stream Operators *****************************
std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
    for (int i = 0; i < m._rows; ++i)
        {
        for (int j = 0; j < m._cols; ++j) {
            if (m(i, j) > PRETTY_PRINT_THRESHOLD){os << "**";}
            else{os << "  ";}
        }
        os << std::endl;
        }
    return os;
}

std::istream& operator>>(std::istream& is, Matrix& m)
{
    for (int i = 0; i < m._rows * m._cols; ++i)
        {
        if (is.peek() == EOF){throw DATA_ERROR;}
        // Try reading as formatted input first
        if (!(is >> m._data[i]))
        {
            // If formatted input fails, try unformatted input
            is.clear();
            is.read(reinterpret_cast<char*>(&m._data[i]), sizeof(float));
        }
        // if (!is.good()){throw DATA_READ_ERROR;}
        if (!is.good()){break;}
        }
    return is;
}
//****************************************************************************