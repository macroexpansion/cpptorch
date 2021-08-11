#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <functional>


torch::Tensor to_tensor(cv::Mat mat) {
    auto size = mat.size();
    torch::Tensor tensor = torch::from_blob(mat.data,
                                            {1, size.height, size.width, mat.channels()},
                                            torch::requires_grad(false).dtype(torch::kFloat32));
    return tensor.permute({0, 3, 1, 2}).contiguous().pin_memory();
}

cv::Mat to_cvmat(torch::Tensor tensor, bool non_blocking=false) {
    tensor = tensor.squeeze(0)
                   .detach()
                   .permute({1, 2, 0})
                   .contiguous()
                   .mul(255)
                   .clamp(0, 255)
                   .to(torch::kU8)
                   .to(torch::kCPU, /*non_blocking=*/non_blocking);

    int height (tensor.size(0)), width (tensor.size(1)), channels (tensor.size(2));
    cv::Mat mat;
    if (channels == 3) {
        mat = cv::Mat(width, height, CV_8UC3, tensor.data_ptr<uchar>());
    } else if (channels == 1) {
        mat = cv::Mat(width, height, CV_8UC1, tensor.data_ptr<uchar>());
    }

    return mat;
}

torch::Tensor stack_tensor(std::vector<cv::Mat> source_mats, std::function<torch::Tensor (cv::Mat)> fn /* = torch::Tensor (*fn)(cv::Mat) */) {
    // cv::imwrite("mat.jpg", source_mats[0]);
    torch::Tensor tensor, batch;
    bool first_iter (true);
    for (cv::Mat mat : source_mats) {
        tensor = fn(mat);
        if (first_iter) {
            batch = tensor;
            first_iter = false;
            continue;
        }
        batch = torch::cat({batch, tensor});
    }
    return batch;
}

// std::string get_image_type(const cv::Mat& img, bool more_info=true) {
//     std::string r;
//     int type = img.type();
//     uchar depth = type & CV_MAT_DEPTH_MASK;
//     uchar chans = 1 + (type >> CV_CN_SHIFT);

//     switch (depth) {
//     case CV_8U:  r = "8U"; break;
//     case CV_8S:  r = "8S"; break;
//     case CV_16U: r = "16U"; break;
//     case CV_16S: r = "16S"; break;
//     case CV_32S: r = "32S"; break;
//     case CV_32F: r = "32F"; break;
//     case CV_64F: r = "64F"; break;
//     default:     r = "User"; break;
//     }

//     r += "C";
//     r += (chans + '0');
   
//     if (more_info)
//         std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

//     return r;
// }

auto to_input(at::Tensor tensor) {
    return std::vector<torch::jit::IValue> {tensor};
}

// auto to_cvimage(at::Tensor tensor) {
//     int width = tensor.sizes()[0];
//     int height = tensor.sizes()[1];
//     try
//     {
//         cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
        
//         show_image(output_mat, "converted image from tensor");
//         return output_mat.clone();
//     }
//     catch (const c10::Error& e)
//     {
//         std::cout << "an error has occured : " << e.msg() << std::endl;
//     }
//     return cv::Mat(height, width, CV_8UC3);
// }

auto threshold(torch::Tensor tensor, float threshold = 0.1) {
    auto zero_indices = tensor < 0.1;
    auto one_indices = tensor >= 0.1;
    tensor.masked_fill_(zero_indices, 0);
    tensor.masked_fill_(one_indices, 1);
    return tensor;
}

std::vector<int> to_int_array(torch::Tensor predictions) {
    predictions = predictions.to(torch::kInt).to(torch::kCPU);
    int* ptr = predictions.data_ptr<int>();

    std::vector<int> results;
    for (size_t i = 0; i < predictions.size(0); i++) {
        results.push_back(*ptr);
        ptr++;
    }
    return results;
}

#endif