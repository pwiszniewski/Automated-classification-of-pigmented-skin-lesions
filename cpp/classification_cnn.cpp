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
  size_t test_batch_size = 32;
  size_t iterations = 30;
  size_t log_interval = 20;
  // path must end in delimiter
  std::string images_path = "../../images/";
  std::string csv_path = "../../data/HAM10000_metadata.csv";
  torch::DeviceType device = torch::kCPU;
};

static Options options;

using Data = std::vector<std::pair<std::string, long>>;

class DatasetFromImages : public torch::data::datasets::Dataset<DatasetFromImages> {
  using Example = torch::data::Example<>;

  Data data;

 public:
  DatasetFromImages(const Data& data) : data(data) {}

  Example get(size_t index) {
    std::string path = options.images_path + data[index].first + ".jpg";
    auto mat = cv::imread(path);
    assert(!mat.empty());

    cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);

    auto R = torch::from_blob(
        channels[2].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto G = torch::from_blob(
        channels[1].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto B = torch::from_blob(
        channels[0].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);

    auto tdata = torch::cat({R, G, B})
                     .view({3, options.image_size, options.image_size})
                     .to(torch::kFloat);
    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
    return {tdata, tlabel};
  }

  torch::optional<size_t> size() const {
    return data.size();
  }
};


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

  std::map<std::string,int> map_label_to_num = {
    {"akiec", 0},
    {"bcc", 1},
    {"bkl", 2},
    {"df", 3},
    {"mel", 4},
    {"nv", 5},
    {"vasc", 6}
  };


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
        path = word;
      }
      else if (nword == 2) {
        label = word;
      }
      nword++;
    }
    nline++;
    data.push_back(std::make_pair(path, map_label_to_num[label]));    
    
  }
  stream.close();

  return data;
}


std::map<long, Data> group_data_by_label(Data data) {
  std::map<long, Data > data_by_label;
  for (const auto & d : data) data_by_label[d.second].push_back(d);
  return data_by_label;
}


void shuffle_data(Data& data) {
  static auto rng = std::default_random_engine {};
  std::shuffle(std::begin(data), std::end(data), rng);
}


Data concatenate_data_map(std::map<long, Data > data_map) {
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

void get_n_first_elements_for_each_key(std::map<long, Data>& data_map, int n) {
  for (const auto& [key, data] : data_map) {
      data_map[key].resize(n);
  }
}


std::pair<Data, Data> train_test_split(Data data, float test_size, bool stratify=false) {
  if (stratify == true) {
    std::map<long, Data> train_data_map, test_data_map;
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


std::pair<Data, Data> prepare_datasets() {
  Data data = read_info();
  shuffle_data(data);
  auto data_by_label = group_data_by_label(data);
  get_n_first_elements_for_each_key(data_by_label, 100);
  data = concatenate_data_map(data_by_label);
  auto train_test_data = train_test_split(data, 0.2, true);
  Data train_data = train_test_data.first;
  Data test_data = train_test_data.second;

  return std::make_pair(train_data, test_data);
}


struct NetworkImpl : torch::nn::SequentialImpl {
  NetworkImpl() {
    using namespace torch::nn;

    auto stride = torch::ExpandingArray<2>({2, 2});
    torch::ExpandingArray<2> shape({-1, 256 * 6 * 6});
    push_back(Conv2d(Conv2dOptions(3, 64, 11).stride(4).padding(2)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Conv2d(Conv2dOptions(64, 192, 5).padding(2)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Conv2d(Conv2dOptions(192, 384, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Conv2d(Conv2dOptions(384, 256, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Conv2d(Conv2dOptions(256, 256, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Functional(torch::reshape, shape));
    push_back(Dropout());
    push_back(Linear(256 * 6 * 6, 4096));
    push_back(Functional(torch::relu));
    push_back(Dropout());
    push_back(Linear(4096, 4096));
    push_back(Functional(torch::relu));
    push_back(Linear(4096, 102));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::nn::functional::detail::log_softmax, 1, torch::nullopt));        
  }
};
TORCH_MODULE(Network);

template <typename DataLoader>
void train(
    Network& network,
    DataLoader& loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t data_size) {
  size_t index = 0;
  network->train();
  float Loss = 0, Acc = 0;

  for (auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();

    if (index++ % options.log_interval == 0) {
      auto end = std::min(data_size, (index + 1) * options.train_batch_size);

      std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
                << std::endl;
    }
  }
}

template <typename DataLoader>
void test(Network& network, DataLoader& loader, size_t data_size) {
  size_t index = 0;
  network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0, Acc = 0;

  for (const auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();
  }

  if (index++ % options.log_interval == 0)
    std::cout << "Test Loss: " << Loss / data_size
              << "\tAcc: " << Acc / data_size << std::endl;
}

int main() {
  torch::manual_seed(1);

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto data = prepare_datasets();

  auto train_set =
      DatasetFromImages(data.first).map(torch::data::transforms::Stack<>());
  auto train_size = train_set.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_set), options.train_batch_size);

  auto test_set =
      DatasetFromImages(data.second).map(torch::data::transforms::Stack<>());
  auto test_size = test_set.size().value();
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_set), options.test_batch_size);

  Network network;
  network->to(options.device);

  torch::optim::SGD optimizer(
      network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

  for (size_t i = 0; i < options.iterations; ++i) {
    train(network, *train_loader, optimizer, i + 1, train_size);
    std::cout << std::endl;
    test(network, *test_loader, test_size);
    std::cout << std::endl;
  }

  std::cout << "END";
  return 0;
}
