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

    // 4. Otsu 二值化，阈值兜底（防止整体偏暗时误判）
    cv::Mat binary;
    double otsuThresh = cv::threshold(small, binary, 0, 255,
                                      cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (otsuThresh < 100.0) {
        // 阈值太低，强行提高到 150，保证纸张被识别为白色
        cv::threshold(small, binary, 150, 255, cv::THRESH_BINARY);
    }

    // 5. 大核闭运算，把文字、条纹并入纸张区域
    cv::Mat closeKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, closeKernel);

    // 6. 找最大白色连通域（纸张）
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    if (numLabels <= 1) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    int maxLabel = -1;
    int maxArea = 0;
    for (int i = 1; i < numLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > maxArea) {
            maxArea = area;
            maxLabel = i;
        }
    }

    if (maxLabel < 0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 7. 只保留最大连通域
    cv::Mat paperMask = (labels == maxLabel);
    paperMask.convertTo(paperMask, CV_8UC1, 255);

    // 8. 找最大轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(paperMask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    double maxContourArea = 0.0;
    int maxIdx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > maxContourArea) {
            maxContourArea = a;
            maxIdx = static_cast<int>(i);
        }
    }

    if (maxIdx < 0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 9. 得到斜矩形边界
    cv::RotatedRect rotRect = cv::minAreaRect(contours[maxIdx]);

    // 10. 面积占比
    const double imgAreaSmall = static_cast<double>(small.cols) * small.rows;
    result.detectedAreaRatio = maxContourArea / imgAreaSmall;

    if (result.detectedAreaRatio < options.minAreaRatio) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 11. 还原坐标到原图
    if (scale != 1.0) {
        rotRect.center.x /= static_cast<float>(scale);
        rotRect.center.y /= static_cast<float>(scale);
        rotRect.size.width  /= static_cast<float>(scale);
        rotRect.size.height /= static_cast<float>(scale);
    }

    // 12. 向外扩展几个像素，避免切割纸张边缘
    const float expand = static_cast<float>(options.expandPixels);
    rotRect.size.width  += expand * 2.0f;
    rotRect.size.height += expand * 2.0f;

    // 13. 构建斜矩形掩膜
    cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
    cv::Point2f vertices[4];
    rotRect.points(vertices);
    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; ++i) {
        pts.push_back(cv::Point(
            static_cast<int>(vertices[i].x),
            static_cast<int>(vertices[i].y)));
    }
    std::vector<std::vector<cv::Point>> polys = { pts };
    cv::fillPoly(mask, polys, cv::Scalar(255));

    // 14. 计算填白区域
    cv::Mat invMask;
    cv::bitwise_not(mask, invMask);
    const int nonZeroCount = cv::countNonZero(invMask);

    if (nonZeroCount < static_cast<int>(static_cast<double>(W) * H * 0.001)) {
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