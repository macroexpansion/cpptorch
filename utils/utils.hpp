#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <tuple>
#include <algorithm>

#include "exceptions/nocontour.hpp"

std::vector<std::vector<cv::Point>> find_contours(cv::Mat mat) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mat, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (contours.size() == 0) throw NoContourException();

    double max(0.0);
    int max_index;
    for (size_t i = 0; i < contours.size(); i++) {
        auto area = cv::contourArea(contours[i]);
        if (area > max) {
            max = area;
            max_index = i;
        }
    }
    return std::vector<std::vector<cv::Point>> {contours[max_index]};
}

void draw_contours(cv::Mat mat,
                   std::vector<std::vector<cv::Point>> contours,
                   std::vector<std::vector<cv::Point>> contours_poly,
                   std::vector<cv::Rect> boundRect) {
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    cv::Scalar color = cv::Scalar(0, 0, 255);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(mat, contours_poly, (int)i, color, 2, cv::LINE_8);
        cv::rectangle(mat, boundRect[i].tl(), boundRect[i].br(), color, 2);
    }
}

std::tuple<std::vector<std::vector<cv::Point>>, std::vector<cv::Rect>> find_bounding_boxes(std::vector<std::vector<cv::Point>> contours) {
    std::vector<std::vector<cv::Point>> contours_poly (contours.size());
    std::vector<cv::Rect> boundRects (contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRects[i] = cv::boundingRect(contours_poly[i]);
    }
    return std::make_tuple(contours_poly, boundRects);
}

std::vector<cv::Mat> crop_rectangles(const cv::Mat source, std::vector<cv::Rect> boundRects, int pad = 20) {
    std::vector<cv::Mat> cropped_mats;
    auto size = source.size();
    for (cv::Rect boundRect : boundRects) {
        cv::Rect paddingRect = boundRect;
        paddingRect.height = std::min(paddingRect.height + 2 * pad, size.height);
        paddingRect.width = std::min(paddingRect.width + 2 * pad, size.width);
        paddingRect.x = std::max(paddingRect.x - pad, 0);
        paddingRect.y = std::max(paddingRect.y - pad, 0);

        cv::Mat crop = source(paddingRect); 
        cv::Mat copyCrop;
        crop.copyTo(copyCrop); // copy data
        cropped_mats.push_back(copyCrop);
    }
    return cropped_mats;
}

#endif