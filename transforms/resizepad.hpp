#ifndef RESIZEPAD_HPP
#define RESIZEPAD_HPP

#include <opencv2/opencv.hpp>
#include <cmath>


int floor_division(auto num) {
    return (int)std::floor(num);
}

cv::Mat resize_pad(cv::Mat image, int size = 256) {
    int height (image.size().height), width (image.size().width);
    cv::Mat resized_mat;
    int top (0), bottom (0), left (0), right (0);

    if (height >= width) {
        cv::resize(image, resized_mat, cv::Size((int)(size * width / height), size), 0, 0, cv::INTER_LINEAR);
        float delta (size - size * width / height);
        left = floor_division(delta / 2);
        right = floor_division(delta / 2);
        // std::printf("%d %d %d %d %.2f \n", top, bottom, left, right, delta);
    } else {
        cv::resize(image, resized_mat, cv::Size(size, (int)(size * height / width)), 0, 0, cv::INTER_LINEAR);
        float delta (size - size * height / width);
        top = floor_division(delta / 2);
        bottom = floor_division(delta / 2);
        // std::printf("%d %d %d %d %.2f \n", top, bottom, left, right, delta);
    }
    cv::Mat bordered_mat, result;
    cv::copyMakeBorder(resized_mat, bordered_mat, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar {0});
    cv::resize(bordered_mat, result, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
    return result;
}

#endif