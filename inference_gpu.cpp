#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <chrono>
#include <fmt/core.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "transforms/fixedsizepad.hpp"
#include "transforms/tensor.hpp"
#include "transforms/resizepad.hpp"
#include "benchmark/benchmark.hpp"
#include "utils/utils.hpp"


using std::cout;
using std::cerr;
using std::endl;

typedef std::chrono::high_resolution_clock::time_point time_point;

const bool non_blocking (false);
const int device_id (1);
const torch::Device device (torch::kCUDA, device_id);
FixedSizePadding tirad_padding(366, 258);
int N_ITER = 0;

torch::Tensor segment_inference(torch::jit::script::Module segment_model, cv::Mat mat) {
    mat.convertTo(mat, CV_32FC3, 1.0 / 255); // !IMPORTANT: need to convert to Float before using to_tensor
    torch::Tensor tensor = to_tensor(mat);
    tensor = tensor.to(device, non_blocking);
    torch::Tensor out = segment_model.forward({tensor}).toTensor();
    cudaDeviceSynchronize();
    return torch::sigmoid(out);
}

torch::Tensor tirad_preprocess(cv::Mat mat) {
    mat.convertTo(mat, CV_32FC3, 1.0 / 255); // !IMPORTANT: need to convert to Float before using to_tensor
    torch::Tensor tensor = to_tensor(mat);
    tensor = tirad_padding->forward(tensor);
    return tensor;
}

torch::Tensor tirad_inference(torch::jit::script::Module tirad_model, std::vector<cv::Mat> source_mats) {
    torch::Tensor batch = stack_tensor(source_mats, tirad_preprocess);
    batch = batch.to(device, non_blocking);
    torch::Tensor out = tirad_model.forward({batch}).toTensor(); 
    cudaDeviceSynchronize();
    return out.argmax(1);
}

torch::Tensor fna_preprocess(cv::Mat mat) {
    mat.convertTo(mat, CV_32FC3, 1.0 / 255); // !IMPORTANT: need to convert to Float before using to_tensor
    cv::Mat resized_mat = resize_pad(mat);
    torch::Tensor tensor = to_tensor(resized_mat);
    tensor = at::upsample_bilinear2d(tensor, {320, 320}, false);
    return tensor;
}

torch::Tensor fna_inference(torch::jit::script::Module fna_model, std::vector<cv::Mat> source_mats) {
    torch::Tensor batch = stack_tensor(source_mats, fna_preprocess);
    batch = batch.to(device, non_blocking);
    torch::Tensor out = fna_model.forward({batch}).toTensor();
    cudaDeviceSynchronize();
    return out.argmax(1);
}

double pipeline(torch::jit::script::Module segment_model,
                torch::jit::script::Module tirad_model,
                torch::jit::script::Module fna_model,
                cv::Mat source) {
    time_point start = std::chrono::high_resolution_clock::now();

    /* segment */
    cv::Mat resized_mat;
    cv::resize(source, resized_mat, cv::Size(352, 352), 0, 0, cv::INTER_LINEAR); // 0.5ms
    torch::Tensor segment = segment_inference(segment_model, resized_mat);
    segment = threshold(segment, 0.1);

    /* resize back to original size */
    cv::Mat segment_mat = to_cvmat(segment);
    cv::Mat original_mat;
    cv::resize(segment_mat, original_mat, source.size(), 0, 0, cv::INTER_LINEAR);
    
    /* find contours */
    try {
        std::vector<std::vector<cv::Point>> contours = find_contours(original_mat);

        if (contours.size() != 0) {
            /* find bounding boxes from contours */
            std::vector<std::vector<cv::Point>> contours_poly;
            std::vector<cv::Rect> boundRects;
            std::tie(contours_poly, boundRects) = find_bounding_boxes(contours);

            /* crop rectangles */
            std::vector<cv::Mat> cropped_mats = crop_rectangles(source, boundRects);

            /* draw contour */
            draw_contours(source, contours, contours_poly, boundRects);

            /* predict tirad */ 
            torch::Tensor tirad_pred = tirad_inference(tirad_model, cropped_mats);
            std::vector<int> tirad_predictions {to_int_array(tirad_pred)};

            /* predict fna */
            torch::Tensor fna_pred = fna_inference(fna_model, cropped_mats);
            std::vector<int> fna_predictions {to_int_array(fna_pred)};

            // cout << "tirad: " << tirad_predictions << ", fna: " << fna_predictions << endl;
            cudaDeviceSynchronize();
        }
    } catch (const NoContourException& e) {
        cerr << e.what() << endl;
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
    }

    /* save draw image */
    // cv::imwrite(fmt::format("saved/saved{}.jpg", N_ITER), source);
    // cout << N_ITER << endl;
    N_ITER++;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    return duration.count();
}

int main(int argc, const char* argv[]) {
    cout << __cplusplus  << '\n';
    cout << "thread num: " << at::get_num_threads() << " | " << at::get_num_interop_threads() << endl;

    cout << "userEnabledCuDNN: " << at::globalContext().userEnabledCuDNN() << endl;
    cout << "userEnabledMkldnn: " << at::globalContext().userEnabledMkldnn() << endl;
    cout << "benchmarkCuDNN: " << at::globalContext().benchmarkCuDNN() << endl;
    cout << "deterministicCuDNN: " << at::globalContext().deterministicCuDNN() << endl;

    if (torch::cuda::is_available()) {
        cout << "Using CUDA: " << device << endl;
    }

    torch::NoGradGuard no_grad;
    torch::jit::script::Module tirad_model, segment_model, fna_model;
    try {
        cudaSetDevice(device_id);

        tirad_model = torch::jit::load("torchscript_module/gpu_traced_tirad_model.pt", device);
        tirad_model.eval();

        segment_model = torch::jit::load("torchscript_module/gpu_traced_segment_model_nas_fpn_hard.pt", device);
        segment_model.eval();
        
        fna_model = torch::jit::load("torchscript_module/gpu_traced_fna_model.pt", device);
        fna_model.eval();

        benchmark_segment(segment_model, 10, device);
        benchmark_fna(fna_model, 10, device);
        benchmark_tirad(tirad_model, 10, device);
    } catch (const c10::Error& e) {
        cerr << "error loading the model\n";
        return -1;
    } catch (const std::exception& e) {
        cerr << e.what() << "\n";
        return -1;
    }

    cv::Mat image = cv::imread("test.BMP", cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // convert opencv image color from BGR -> RGB

    double sum(0.0);
    int num_iter (110), offset (10);
    for (size_t i = 0; i < num_iter; i++) {
        auto time = pipeline(segment_model, tirad_model, fna_model, image.clone());
        if (i < offset) continue;
        sum += time;
    }
    cout << "avg: " << sum / (num_iter - offset) << endl;
}
