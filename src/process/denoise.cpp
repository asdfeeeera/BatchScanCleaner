#include "denoise.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace process {

double Denoise::estimatePaperGray(const cv::Mat &gray)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int mx = static_cast<int>(W * 0.25);
    const int my = static_cast<int>(H * 0.25);

    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(W) * H / 8);
    for (int y = my; y < H - my; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = mx; x < W - mx; x += 2) {
            pixels.push_back(row[x]);
        }
    }
    if (pixels.empty()) {
        cv::Scalar m = cv::mean(gray);
        return m[0];
    }
    const size_t idx = pixels.size() * 9 / 10;
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

void Denoise::buildProtectMask(const cv::Mat &gray,
                                cv::Mat &protectMask,
                                int protectRadius)
{
    // 二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    protectMask = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);

    // ★ 核心改动：保护所有面积 >= 15 的暗块
    //    文字、表格线、印章、签名 面积都远超 15
    const int protectArea = 15;
    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= protectArea) {
            cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                       stats.at<int>(i, cv::CC_STAT_TOP),
                       stats.at<int>(i, cv::CC_STAT_WIDTH),
                       stats.at<int>(i, cv::CC_STAT_HEIGHT));
            cv::rectangle(protectMask, r, cv::Scalar(255), cv::FILLED);
        }
    }

    // 膨胀保护掩膜
    if (protectRadius > 0) {
        const int ksize = protectRadius * 2 + 1;
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::dilate(protectMask, protectMask, kernel);
    }
}

DenoiseResult Denoise::removeSpots(const cv::Mat &src,
                                    const DenoiseOptions &options)
{
    DenoiseResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    const double paperGray = estimatePaperGray(gray);
    const double darkThreshold = paperGray * options.darkRatio;

    // 二值化
    cv::Mat binary;
    cv::threshold(gray, binary, darkThreshold, 255, cv::THRESH_BINARY_INV);

    // 保护掩膜（先把文字等大块保护起来）
    cv::Mat protectMask;
    buildProtectMask(gray, protectMask, options.protectRadius);

    // 连通域分析
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    // 强度对应的最大污点面积
    int maxSpotArea = 30;
    if (options.strengthLevel == 0) maxSpotArea = 15;
    else if (options.strengthLevel == 1) maxSpotArea = 30;
    else if (options.strengthLevel == 2) maxSpotArea = 60;

    cv::Mat spotMask = cv::Mat::zeros(H, W, CV_8UC1);
    int spotCount = 0;
    int totalPixels = 0;

    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);

        // 面积过滤：太大当文字，太小当噪点忽略
        if (area < 2) continue;
        if (area > maxSpotArea) continue;

        // 长宽比过滤：太细长当线条
        const double aspect = static_cast<double>(std::max(w, h)) / std::max(1, std::min(w, h));
        if (aspect > 5.0) continue;

        // ★ 核心改动：如果在保护掩膜内，跳过
        cv::Rect r(x, y, w, h);
        r &= cv::Rect(0, 0, W, H);
        cv::Mat roi = protectMask(r);
        if (cv::countNonZero(roi) > 0) {
            continue;
        }

        // 是污点
        cv::rectangle(spotMask, r, cv::Scalar(255), cv::FILLED);
        ++spotCount;
        totalPixels += area;
    }

    if (spotCount == 0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 轻微膨胀污点掩膜
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(spotMask, spotMask, kernel);

    // ★ 核心改动：膨胀后再减去保护区域，避免 inpaint 破坏文字
    cv::bitwise_and(spotMask, ~protectMask, spotMask);

    // 修补
    cv::Mat dst;
    if (options.useInpaint) {
        cv::inpaint(src, spotMask, dst, 3, cv::INPAINT_TELEA);
    } else {
        cv::medianBlur(src, dst, 3);
        src.copyTo(dst, ~spotMask);
    }

    result.image = dst;
    result.spotCount = spotCount;
    result.cleanedPixels = totalPixels;
    result.ok = true;
    return result;
}

} // namespace process