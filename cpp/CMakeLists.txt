cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(classification_cnn)

find_package(Torch REQUIRED)

add_executable(classification_cnn classification_cnn.cpp)
target_compile_features(classification_cnn PUBLIC cxx_range_for)
target_link_libraries(classification_cnn ${TORCH_LIBRARIES})
