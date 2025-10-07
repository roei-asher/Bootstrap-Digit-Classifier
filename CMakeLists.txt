cmake_minimum_required(VERSION 3.28)
project(ex4)

set(CMAKE_CXX_STANDARD 11)

add_executable(ex4 main.cpp
        Matrix.cpp
        Activation.cpp
        Dense.cpp
        MlpNetwork.cpp)
