#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

BlackEdgeResult BlackEdge::removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options)
{
    BlackEdgeResult result;
    if (src.empty()) {
        return result;
    }

    const int W = src.cols;
    const int H = src.rows;

    // 1. 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    // 2. 高斯模糊
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 3. 降采样，加速检测
    cv::Mat detectImage;
    double scale = 1.0;
    const int maxDim = std::max(W, H);
    if (maxDim > 1500) {
        scale = 1500.0 / maxDim;
        cv::resize(blurred, detectImage, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        detectImage = blurred;
    }

    // 4. 自适应二值化：纸张白，黑边黑
    cv::Mat binary;
    cv::adaptiveThreshold(detectImage, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY,
                          options.blockSize,
                          options.adaptiveC);

    // 5. 形态学处理
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(options.morphSize, options.morphSize));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    // 6. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 7. 找最大轮廓（纸张）
    double maxArea = 0.0;
    int maxIdx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = static_cast<int>(i);
        }
    }

    if (maxIdx < 0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 8. 面积检查
    const double totalAreaDetect = static_cast<double>(detectImage.cols) * detectImage.rows;
    const double areaRatio = maxArea / totalAreaDetect;
    result.detectedAreaRatio = areaRatio;

    if (areaRatio < options.minAreaRatio) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 9. 多边形近似
    std::vector<cv::Point> paperContour = contours[maxIdx];
    const double peri = cv::arcLength(paperContour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(paperContour, approx, peri * options.approxEpsilonRatio, true);

    if (approx.size() < 3) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 10. 坐标还原到原图尺寸
    if (scale != 1.0) {
        for (auto &pt : approx) {
            pt.x = static_cast<int>(pt.x / scale);
            pt.y = static_cast<int>(pt.y / scale);
        }
    }

    // 11. 检查是否贴合图像边缘
    cv::Rect paperBounding = cv::boundingRect(approx);
    const int marginX = static_cast<int>(W * options.edgeMarginRatio);
    const int marginY = static_cast<int>(H * options.edgeMarginRatio);

    const bool touchesLeft = paperBounding.x <= marginX;
    const bool touchesRight = (paperBounding.x + paperBounding.width) >= (W - marginX);
    const bool touchesTop = paperBounding.y <= marginY;
    const bool touchesBottom = (paperBounding.y + paperBounding.height) >= (H - marginY);

    if (touchesLeft && touchesRight && touchesTop && touchesBottom) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 12. 构建掩膜
    cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
    std::vector<std::vector<cv::Point>> paperPoly = { approx };
    cv::fillPoly(mask, paperPoly, cv::Scalar(255));

    // 13. 膨胀掩膜
    if (options.expandPixels > 0) {
        const int ksize = options.expandPixels * 2 + 1;
        cv::Mat expandKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                         cv::Size(ksize, ksize));
        cv::dilate(mask, mask, expandKernel);
    }

    // 14. 计算需填白区域
    cv::Mat invMask;
    cv::bitwise_not(mask, invMask);
    const int nonZeroCount = cv::countNonZero(invMask);

    if (nonZeroCount < static_cast<int>(static_cast<double>(W) * H * 0.005)) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 15. 填白
    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (dst.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }
    dst.setTo(white, invMask);

    result.image = dst;
    result.filledPixels = nonZeroCount;
    result.ok = true;
    return result;
}

} // namespace process