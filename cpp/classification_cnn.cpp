#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Options {
  int image_size = 224;
  size_t train_batch_size = 8;
  size_t test_batch_size = 200;
  size_t iterations = 10;
  size_t log_interval = 20;
  // path must end in delimiter
  std::string images_path = "../../images/";
  std::string csv_path = "../../data/HAM10000_metadata.csv";
  // std::string datasetPath = "./dataset/";
  // std::string infoFilePath = "info.txt";
  torch::DeviceType device = torch::kCPU;
};

static Options options;

using Data = std::vector<std::pair<std::string, std::string>>;

std::pair<Data, Data> readInfo() {
  Data train, test;

  std::ifstream stream(options.csv_path);
  assert(stream.is_open());

  std::string path, label;
  std::string buffer, word;
  int nline = 0;
  while (!stream.eof()) {
    getline(stream, buffer, '\n');
    if (nline == 0) {
      nline++;
      continue; 
    }
    std::string temp;
    std::stringstream ss(buffer); 
    int nword = 0;
    while (std::getline(ss, word, ',')) { 
      if (nword == 1) {
        path = options.images_path + word;
      }
      else if (nword == 2) {
        label = word;
      }
      nword++;
    }
    nline++;
    train.push_back(std::make_pair(path, label));    
    
  }
  stream.close();
  for (int i = 0; i < train.size(); i++) {
    if (i < 10 or i > 10010)
      std::cout << i << " " << train[i].first << " " << train[i].second << std::endl;
  }
  
  return std::make_pair(train, test);
}

int main() {
  torch::manual_seed(1);

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto data = readInfo();

  std::cout << "END";
  return 0;
}
