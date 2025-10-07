# Handwritten Digit Classifier (Neural Network from Scratch)

A C++ implementation of a fully-connected neural network for handwritten digit recognition (0-9), built entirely from scratch without using external machine learning libraries. This project was completed as part of the C & C++ Programming Workshop course (67315) at The Hebrew University of Jerusalem.

## Overview

This neural network achieves **~96% accuracy** on handwritten digit classification by implementing a 4-layer fully-connected architecture. The project demonstrates core concepts in:
- **Object-Oriented Programming** in C++
- **Linear Algebra** operations (matrix multiplication, vectorization, transposition)
- **Neural Network fundamentals** (forward propagation, activation functions)
- **Operator overloading** for intuitive matrix operations
- **Memory management** (Rule of Three, dynamic allocation)
- **Exception handling**

## Architecture

The network consists of 4 dense layers with the following configuration:

| Layer | Input Size | Output Size | Weights Dimensions | Bias Dimensions | Activation Function |
|-------|-----------|-------------|-------------------|-----------------|---------------------|
| 1     | 784       | 128         | 128×784          | 128×1          | ReLU                |
| 2     | 128       | 64          | 64×128           | 64×1           | ReLU                |
| 3     | 64        | 20          | 20×64            | 20×1           | ReLU                |
| 4     | 20        | 10          | 10×20            | 10×1           | Softmax             |

### Network Flow

```
Input (28×28 image) → Vectorize to 784×1
  ↓
Layer 1: (W₁ · x + b₁) → ReLU → 128×1
  ↓
Layer 2: (W₂ · r₁ + b₂) → ReLU → 64×1
  ↓
Layer 3: (W₃ · r₂ + b₃) → ReLU → 20×1
  ↓
Layer 4: (W₄ · r₃ + b₄) → Softmax → 10×1
  ↓
Output: Probability distribution over digits 0-9
```

### Activation Functions

- **ReLU (Rectified Linear Unit)**:
  ```
  ReLU(x) = max(0, x)
  ```
  Applied element-wise to introduce non-linearity.

- **Softmax**:
  ```
  softmax(x)ᵢ = e^(xᵢ) / Σⱼ(e^(xⱼ))
  ```
  Converts the final layer output into a probability distribution.

## Core Components

### 1. Matrix Class (`Matrix.h`, `Matrix.cpp`)
A comprehensive matrix implementation with:
- **Constructors**: Default, parameterized, copy constructor
- **Destructor**: Proper memory cleanup
- **Operators**:
  - Arithmetic: `+`, `+=`, `*` (matrix multiplication and scalar)
  - Access: `()` (row, col indexing), `[]` (linear indexing)
  - Stream: `<<`, `>>` (for I/O operations)
- **Methods**:
  - `transpose()`: Matrix transposition
  - `vectorize()`: Flatten to column vector (row-major order)
  - `dot()`: Element-wise (Hadamard) product
  - `norm()`: Frobenius norm calculation
  - `rref()`: Reduced Row Echelon Form
  - `argmax()`: Find index of maximum element
  - `sum()`: Sum of all elements

### 2. Activation Functions (`Activation.h`, `Activation.cpp`)
Implemented in the `activation` namespace:
- `activation::relu(Matrix)`: ReLU activation
- `activation::softmax(Matrix)`: Softmax activation

Both functions operate on entire matrices, not just column vectors.

### 3. Dense Layer (`Dense.h`, `Dense.cpp`)
Represents a single neural network layer:
```cpp
Dense layer(weights, bias, activation_function);
Matrix output = layer(input);  // Applies: activation(W·x + b)
```

### 4. MLP Network (`MlpNetwork.h`, `MlpNetwork.cpp`)
The main neural network class that chains 4 `Dense` layers:
```cpp
MlpNetwork mlp(weights_array, biases_array);
digit result = mlp(input_image);  // Returns digit struct with value and probability
```

## Input/Output Format

### Input
- **Images**: 28×28 grayscale images stored as binary files
- Each pixel is a `float` value between 0 and 1
- Images are vectorized to 784×1 before processing

### Output
The network returns a `digit` struct containing:
- `value`: The predicted digit (0-9)
- `probability`: Confidence of the prediction (0-1)

Example output:
```
Image processed:
[ASCII art representation of the digit]
Mlp result: 7 at probability: 0.90
```

## Building and Running

### Prerequisites
- C++ compiler with C++11 support or later
- Make
- Python 3.x (for visualization tool)

### Setup

#### 1. Python Environment (Optional - for visualization)
To use the `plot_img.py` visualization tool, set up a Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Compilation
```bash
make mlpnetwork
```

### Usage

#### Running the Classifier
```bash
./mlpnetwork parameters/w1 parameters/w2 parameters/w3 parameters/w4 parameters/b1 parameters/b2 parameters/b3 parameters/b4
```

Or using the shorthand if running from the project root:
```bash
./mlpnetwork parameters/w{1..4} parameters/b{1..4}
```

#### Visualizing Images (Optional)
To view what the binary image files look like:
```bash
python plot_img.py images/im3
```

### Interactive Mode
Once running, the program prompts for image paths:
```
Please insert image path:
> images/im3
Image processed:
[ASCII art representation of the digit]
Mlp result: 3 at probability: 0.98

Please insert image path:
> images/im7
Image processed:
[...]
Mlp result: 7 at probability: 0.92

Please insert image path:
> q
```
Enter `q` to quit.

## Technical Highlights

### Memory Management
- Implements the **Rule of Three** (copy constructor, assignment operator, destructor)
- Dynamic memory allocation using `new`/`delete`
- No memory leaks (verified with Valgrind)

### Operator Overloading
Intuitive matrix operations:
```cpp
Matrix A(3, 3), B(3, 3);
Matrix C = A + B;           // Matrix addition
Matrix D = A * B;           // Matrix multiplication
Matrix E = 2.0 * A;         // Scalar multiplication
float val = A(0, 0);        // Element access
float val2 = A[4];          // Linear indexing
```

### Error Handling
- Exception handling for invalid inputs
- Dimension mismatch detection
- File I/O error handling

### Numerical Considerations
- Uses `float` for matrix elements
- Matrix multiplication follows standard mathematical order to minimize floating-point errors
- Tolerates small numerical deviations (±0.001) in RREF computation

## File Structure

```
.
├── Matrix.h / Matrix.cpp          # Matrix class implementation
├── Activation.h / Activation.cpp  # Activation functions
├── Dense.h / Dense.cpp            # Dense layer implementation
├── MlpNetwork.h / MlpNetwork.cpp  # Neural network class
├── main.cpp                       # CLI interface
├── Makefile                       # Build configuration
├── CMakeLists.txt                 # CMake build configuration
├── plot_img.py                    # Python utility to visualize binary images
├── requirements.txt               # Python dependencies
├── parameters/                    # Pre-trained weights and biases
│   ├── w1, w2, w3, w4            # Weight matrices
│   └── b1, b2, b3, b4            # Bias vectors
└── images/                        # Sample digit images
    └── im0, im1, ..., im9        # Test images
```

## Learning Outcomes

This project demonstrates proficiency in:
1. **C++ fundamentals**: Classes, operator overloading, const-correctness
2. **Memory management**: Dynamic allocation, Rule of Three
3. **Linear algebra**: Matrix operations from scratch
4. **Neural networks**: Understanding forward propagation without frameworks
5. **Software engineering**: Modular design, error handling, documentation

## Performance

- **Accuracy**: ~96% on handwritten digit classification
- **Network size**: 4 layers with 101,770 trainable parameters
- **Prediction**: Near-instantaneous on modern hardware

## Limitations & Notes

- This is an **inference-only** implementation (no training code)
- Pre-trained weights are provided as binary files
- No STL containers used (per assignment requirements)
- Manual memory management throughout

## Acknowledgments

Developed as part of the C & C++ Programming Workshop (Course 67315) at The Hebrew University of Jerusalem, School of Engineering and Computer Science.

## License

This is an academic project for educational purposes.
