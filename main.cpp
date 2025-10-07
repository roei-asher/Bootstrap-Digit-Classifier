// main.cpp
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "Matrix.h"
#include "Activation.h"
#include "Dense.h"
#include "MlpNetwork.h"

#define QUIT "q"
#define INSERT_IMAGE_PATH "Please insert image path:"
#define ERROR_INVALID_PARAMETER "Error: invalid Parameters file for layer: "
#define ERROR_INVALID_INPUT "Error: Failed to retrieve input. Exiting.."
#define ERROR_INVALID_IMG "Error: invalid image path or size: "
#define USAGE_MSG "Usage:\n" \
                  "\t./mlpnetwork w1 w2 w3 w4 b1 b2 b3 b4\n" \
                  "\twi - the i'th layer's weights\n" \
                  "\tbi - the i'th layer's biases"
#define USAGE_ERR "Error: wrong number of arguments."
#define ARGS_START_IDX 1
#define ARGS_COUNT (ARGS_START_IDX + (MLP_SIZE * 2))
#define WEIGHTS_START_IDX ARGS_START_IDX
#define BIAS_START_IDX (ARGS_START_IDX + MLP_SIZE)

void usage(int argc) {
    if (argc != ARGS_COUNT) {
        throw std::domain_error(USAGE_ERR);
    }
    std::cout << USAGE_MSG << std::endl;
}

bool readFileToMatrix(const std::string &filePath, Matrix &mat) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        return false;
    }

    float* data = new float[mat.get_rows() * mat.get_cols()];
    file.read(reinterpret_cast<char*>(data), mat.get_rows() * mat.get_cols() * sizeof(float));

    if (file.fail()) {
        delete[] data;
        return false;
    }

    for (int i = 0; i < mat.get_rows(); ++i) {
        for (int j = 0; j < mat.get_cols(); ++j) {
            mat(i, j) = data[i * mat.get_cols() + j];
        }
    }

    delete[] data;
    return true;
}

void loadParameters(char *paths[ARGS_COUNT], Matrix weights[MLP_SIZE],
                    Matrix biases[MLP_SIZE]) {
    for (int i = 0; i < MLP_SIZE; i++) {
        weights[i] = Matrix(weights_dims[i].rows, weights_dims[i].cols);
        biases[i] = Matrix(bias_dims[i].rows, bias_dims[i].cols);

        std::string weightsPath(paths[WEIGHTS_START_IDX + i]);
        std::string biasPath(paths[BIAS_START_IDX + i]);

        if (!(readFileToMatrix(weightsPath, weights[i]) &&
              readFileToMatrix(biasPath, biases[i]))) {
            auto msg = ERROR_INVALID_PARAMETER + std::to_string(i + 1);
            throw std::invalid_argument(msg);
        }
    }
}

void mlpCli(MlpNetwork &mlp) {
    Matrix img(img_dims.rows, img_dims.cols);
    std::string imgPath;

    std::cout << INSERT_IMAGE_PATH << std::endl;
    std::cin >> imgPath;
    if (!std::cin.good()) {
        throw std::invalid_argument(ERROR_INVALID_INPUT);
    }

    while (imgPath != QUIT) {
        if (readFileToMatrix(imgPath, img)) {
            Matrix imgVec = img;
            digit output = mlp(imgVec.vectorize());
            std::cout << "Image processed:" << std::endl
                      << img << std::endl;
            std::cout << "Mlp result: " << output.value <<
                      " at probability: " << output.probability << std::endl;
        } else {
            throw std::invalid_argument(ERROR_INVALID_IMG + imgPath);
        }

        std::cout << INSERT_IMAGE_PATH << std::endl;
        std::cin >> imgPath;
        if (!std::cin.good()) {
            throw std::invalid_argument(ERROR_INVALID_INPUT);
        }
    }
}

int main(int argc, char **argv) {
    try {
        usage(argc);
    } catch (const std::domain_error &domainError) {
        std::cerr << domainError.what() << std::endl;
        return EXIT_FAILURE;
    }

    Matrix weights[MLP_SIZE];
    Matrix biases[MLP_SIZE];

    try {
        loadParameters(argv, weights, biases);
    } catch (const std::invalid_argument &invalidArgument) {
        std::cerr << invalidArgument.what() << std::endl;
        return EXIT_FAILURE;
    }

    MlpNetwork mlp(weights, biases);

    try {
        mlpCli(mlp);
    } catch (const std::invalid_argument &invalidArgument) {
        std::cerr << invalidArgument.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}