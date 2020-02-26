#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

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

void print_data(const Data& data) {
  for (auto& d : data) {
    std::cout << d.first << " " << d.second << std::endl;
  }
}

Data read_info() {
  Data data;

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
    data.push_back(std::make_pair(path, label));    
    
  }
  stream.close();

  return data;
}


std::map<std::string, Data> group_data_by_label(Data data) {
  std::map<std::string, Data > data_by_label;
  for (const auto & d : data) data_by_label[d.second].push_back(d);
  return data_by_label;
}


void shuffle_data(Data& data) {
  static auto rng = std::default_random_engine {};
  std::shuffle(std::begin(data), std::end(data), rng);
}


Data concatenate_data_map(std::map<std::string, Data > data_map) {
  Data data;
  for (const auto& [key, d] : data_map) {
    data.insert(
      data.end(),
      std::make_move_iterator(d.begin()),
      std::make_move_iterator(d.end())
    );
  }
  return data;
}

void get_n_first_elements_for_each_key(std::map<std::string, Data>& data_map, int n) {
  for (const auto& [key, data] : data_map) {
      data_map[key].resize(n);
  }
}


std::pair<Data, Data> train_test_split(Data data, float test_size, bool stratify=false) {
  if (stratify == true) {
    std::map<std::string, Data> train_data_map, test_data_map;
    auto data_by_label = group_data_by_label(data);
    for (const auto& [key, d] : data_by_label) {
      std::size_t const split_size = d.size() * test_size;
      Data train_data(d.begin() + split_size, d.end());
      Data test_data(d.begin(), d.begin() + split_size);
      train_data_map[key] = train_data;
      test_data_map[key] = test_data;
    }
    Data train_data = concatenate_data_map(train_data_map);
    Data test_data = concatenate_data_map(test_data_map);
    shuffle_data(train_data);
    shuffle_data(test_data);
    return std::make_pair(train_data, test_data); 
  }
  else {
    std::size_t const split_size = data.size() * test_size;
    Data train_data(data.begin() + split_size, data.end());
    Data test_data(data.begin(), data.begin() + split_size);
    return std::make_pair(train_data, test_data); 
  }
}


void prepare_datasets() {
  Data data = read_info();
  shuffle_data(data);
  auto data_by_label = group_data_by_label(data);
  get_n_first_elements_for_each_key(data_by_label, 10);
  data = concatenate_data_map(data_by_label);
  auto train_test_data = train_test_split(data, 0.2, true);
  Data train_data = train_test_data.first;
  Data test_data = train_test_data.second;
}


int main() {
  torch::manual_seed(1);

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  prepare_datasets();

  std::cout << "END";
  return 0;
}
