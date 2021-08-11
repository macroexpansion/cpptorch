#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>
#include <fmt/core.h>

#include "transforms/fixedsizepad.hpp"
#include "transforms/tensor.hpp"

auto inference(torch::jit::script::Module model, std::vector<torch::jit::IValue> inputs) {
    auto start = std::chrono::high_resolution_clock::now();

    auto out = model.forward(inputs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    // std::cout << "time: " << duration.count() << "s" << std::endl;
    return duration.count();
}

void benchmark_tirad(torch::jit::Module model) {
    float average = 0.0;
    for (int i = 0; i < 102; i++) {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor image = torch::ones({1, 3, 258, 366});
        inputs.push_back(image);
        auto time = inference(model, inputs);

        if (i >= 2) {
            average += time;
        }
    }
    std::cout << average / 100 << std::endl;
}

void benchmark_segment(torch::jit::Module model) {
    float average = 0.0;
    for (int i = 0; i < 102; i++) {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor image = torch::ones({1, 3, 352, 352});
        inputs.push_back(image);
        auto time = inference(model, inputs);
        
        if (i >= 2) {
            average += time;
        }
    } 
    std::cout << average / 100 << std::endl;
}

void benchmark_fna(torch::jit::Module model) {
    auto average = 0.0;
    for (int i = 0; i < 102; i++) {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor image = torch::ones({1, 3, 288, 384});
        inputs.push_back(image);
        auto time = inference(model, inputs);

        if (i >= 2) {
            average += time;
        }
    }
    std::cout << average / 100 << std::endl;
}

// void test_imread() {
//     cv::Mat image = cv::imread("test.jpg", cv::IMREAD_COLOR);
//     auto tensor = cvmat_2_tensor(image);
//     auto mat = tensor_2_cvmat(tensor, true);
//     std::cout << mat.size() << " " << mat.channels() << std::endl;

//     cv::imshow("show", mat);
//     cv::waitKey(0);
// }

int main(int argc, const char* argv[]) {
    // at::init_num_threads();
    // std::cout << "thread num: " << at::get_num_threads() << " | " << at::get_num_interop_threads() <<std::endl;

    // torch::jit::script::Module tirad_model, segment_model, fna_model;
    // try {
    //    tirad_model = torch::jit::load("torchscript_module/traced_tirad_model.pt");
    //    tirad_model.eval();

    //    segment_model = torch::jit::load("torchscript_module/traced_segment_model.pt");
    //    segment_model.eval();
       
    //    fna_model = torch::jit::load("torchscript_module/traced_fna_model.pt");
    //    fna_model.eval();
    // } catch (const c10::Error& e) {
    //     std::cerr << "error loading the model\n";
    //     return -1;
    // }
    
    // benchmark_tirad(tirad_model);
    // benchmark_segment(segment_model);
    // benchmark_fna(fna_model);

    // test_imread();
    
    torch::Tensor tensor = torch::ones({1,1,3,3});
    float threshold = 1;
    auto zero_indices = tensor < threshold;
    auto one_indices = tensor >= threshold;
    tensor.masked_fill_(zero_indices, 0);
    tensor.masked_fill_(one_indices, 2);
    std::cout << torch::max(tensor) << std::endl;
    std::cout << tensor.index({0}) << std::endl;
}