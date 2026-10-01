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

    // 2. 降采样加速
    cv::Mat small;
    double scale = 1.0;
    const int maxDim = std::max(W, H);
    if (maxDim > 1200) {
        scale = 1200.0 / maxDim;
        cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        small = gray;
    }

    // 3. 高斯模糊
    cv::GaussianBlur(small, small, cv::Size(5, 5), 0);

    // 4. 二值化：Otsu，但如果阈值太低（图整体偏暗）用固定 200
    cv::Mat binary;
    double otsuThresh = cv::threshold(small, binary, 0, 255,
                                      cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (otsuThresh < 180.0) {
        cv::threshold(small, binary, 200, 255, cv::THRESH_BINARY);
    }

    // 5. 轻量开运算去噪（3×3，不会吞掉黑边）
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    // 6. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

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

    // 8. 面积占比
    const double imgAreaSmall = static_cast<double>(small.cols) * small.rows;
    result.detectedAreaRatio = maxArea / imgAreaSmall;

    if (result.detectedAreaRatio < options.minAreaRatio) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 9. 多边形近似：让纸张轮廓更平滑、贴边
    std::vector<cv::Point> paperContour = contours[maxIdx];
    const double peri = cv::arcLength(paperContour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(paperContour, approx, peri * 0.001, true);

    if (approx.size() < 3) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 10. 还原坐标到原图
    if (scale != 1.0) {
        for (auto &pt : approx) {
            pt.x = static_cast<int>(pt.x / scale);
            pt.y = static_cast<int>(pt.y / scale);
        }
    }

    // 11. 构建掩膜：多边形 = 纸张区域
    cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
    std::vector<std::vector<cv::Point>> polys = { approx };
    cv::fillPoly(mask, polys, cv::Scalar(255));

    // 12. 掩膜膨胀：向外扩展几个像素，覆盖黑边
    if (options.expandPixels > 0) {
        const int ksize = options.expandPixels * 2 + 1;
        cv::Mat expandKernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::dilate(mask, mask, expandKernel);
    }

    // 13. 计算掩膜外区域
    cv::Mat invMask;
    cv::bitwise_not(mask, invMask);
    const int nonZeroCount = cv::countNonZero(invMask);

    if (nonZeroCount < static_cast<int>(static_cast<double>(W) * H * 0.001)) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 14. 掩膜外填白
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