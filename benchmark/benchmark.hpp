#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <chrono>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>


auto inference(torch::jit::script::Module model, std::vector<torch::jit::IValue> inputs) {
    auto start = std::chrono::high_resolution_clock::now();

    auto out = model.forward(inputs).toTensor();
    // cudatorch::kCUDASynchronize();
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    // std::cout << duration.count() << "s" << std::endl;
    return duration.count();
}

auto benchmark_tirad(torch::jit::Module model, int num=5, torch::Device device=torch::kCPU) {
    double sum = 0.0;
    for (size_t i = 0; i < num; i++) {
        torch::Tensor image = torch::ones({1, 3, 258, 366});
        image.fill_(100);
        image = image.to(device);
        std::vector<torch::jit::IValue> inputs {image};
        auto time = inference(model, inputs);
        if (i < 2) continue;
        sum += time;
    }
    std::cout << "tirad: " << sum / (num - 2) << "s" << std::endl;
    return sum / (num - 2);
}

auto benchmark_segment(torch::jit::Module model, int num=5, torch::Device device=torch::kCPU) {
    double sum = 0.0;
    for (size_t i = 0; i < num; i++) {
        torch::Tensor image = torch::ones({1, 3, 352, 352});
        image = image.to(device);
        std::vector<torch::jit::IValue> inputs {image};
        auto time = inference(model, inputs);
        if (i < 2) continue;
        sum += time;
    } 
    std::cout << "segment: " << sum / (num - 2) << "s" << std::endl;
    return sum / (num - 2);
}

auto benchmark_fna(torch::jit::Module model, int num=5, torch::Device device=torch::kCPU) {
    double sum = 0.0;
    for (size_t i = 0; i < num; i++) {
        torch::Tensor image = torch::ones({1, 3, 320, 320});
        image = image.to(device);
        std::vector<torch::jit::IValue> inputs {image};
        auto time = inference(model, inputs);
        if (i < 2) continue;
        sum += time;
    }
    std::cout << "fna: " << sum / (num - 2) << "s" << std::endl;
    return sum / (num - 2);
}

void benchmark(torch::jit::script::Module tirad_model,
               torch::jit::script::Module segment_model,
               torch::jit::script::Module fna_model,
               torch::Device device) {
    double tirad(0.0), segment(0.0), fna(0.0); 
    tirad = benchmark_tirad(tirad_model, 102, device);
    segment = benchmark_segment(segment_model, 102, device);
    fna = benchmark_fna(fna_model, 102, device);
    std::cout << "sum: " << tirad + segment + fna << "s" << std::endl;
}

#endif