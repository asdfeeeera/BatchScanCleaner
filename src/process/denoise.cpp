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
    const int W = gray.cols;
    const int H = gray.rows;

    protectMask = cv::Mat::zeros(H, W, CV_8UC1);

    // 1. 检测"细长结构"（文字笔画、表格线、下划线）作为保护对象
    //    方法：形态学开运算，提取水平/垂直/对角长条
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 水平长条
    cv::Mat hKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 1));
    cv::Mat hLines;
    cv::morphologyEx(binary, hLines, cv::MORPH_OPEN, hKernel);

    // 垂直长条
    cv::Mat vKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 15));
    cv::Mat vLines;
    cv::morphologyEx(binary, vLines, cv::MORPH_OPEN, vKernel);

    // 合并
    cv::bitwise_or(hLines, vLines, protectMask);

    // 2. 文字笔画：任何小连通域也保护（标点、小数点）
    //    取所有"面积小但有一定结构"的连通域
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // 标点/句号/小数点：面积小（10-80），宽高比正常
        const bool isPunctuation =
            (area >= 10 && area <= 80) &&
            (w <= 12 && h <= 12) &&
            (w >= 2 && h >= 2);

        // 文字笔画：面积中等，一个方向较细
        const bool isTextStroke =
            (area >= 20 && area <= 500) &&
            (w <= 60 && h <= 60) &&
            (std::min(w, h) <= 5);

        // 表格线交叉点、印章边框、签名笔画
        const bool isStructure =
            (area >= 50 && area <= 2000) &&
            (std::max(w, h) >= 20);

        if (isPunctuation || isTextStroke || isStructure) {
            cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                       stats.at<int>(i, cv::CC_STAT_TOP),
                       w, h);
            cv::rectangle(protectMask, r, cv::Scalar(255), cv::FILLED);
        }
    }

    // 3. 膨胀保护掩膜
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

    // 1. 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    // 2. 估算纸张灰度
    const double paperGray = estimatePaperGray(gray);
    const double darkThreshold = paperGray * options.darkRatio;

    // 3. 二值化：暗块 = 前景
    cv::Mat binary;
    cv::threshold(gray, binary, darkThreshold, 255, cv::THRESH_BINARY_INV);

    // 4. 构建保护掩膜
    cv::Mat protectMask;
    buildProtectMask(gray, protectMask, options.protectRadius);

    // 5. 清除保护区域的暗块
    cv::Mat candidate;
    cv::bitwise_and(binary, ~protectMask, candidate);

    // 6. 根据强度调整面积阈值
    int maxArea = options.maxSpotArea;
    if (options.strengthLevel == 0) {
        maxArea = maxArea / 2;   // 保守
    } else if (options.strengthLevel == 2) {
        maxArea = maxArea * 2;   // 强力
    }

    // 7. 连通域分析，识别污点
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        candidate, labels, stats, centroids, 8, CV_32S);

    // 8. 构建污点掩膜
    cv::Mat spotMask = cv::Mat::zeros(H, W, CV_8UC1);
    int spotCount = 0;
    int totalPixels = 0;

    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // 面积过滤
        if (area > maxArea) continue;
        if (area < 3) continue;  // 太小的忽略（噪点级）

        // 长宽过滤
        if (w > options.maxSpotWidth) continue;
        if (h > options.maxSpotHeight) continue;

        // 长宽比过滤：排除细长结构（可能是线条）
        const double aspect = static_cast<double>(std::max(w, h)) / std::max(1, std::min(w, h));
        if (aspect > 8.0) continue;

        // 是污点
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);

        cv::Rect r(x, y, w, h);
        r &= cv::Rect(0, 0, W, H);
        cv::rectangle(spotMask, r, cv::Scalar(255), cv::FILLED);

        ++spotCount;
        totalPixels += area;
    }

    // 9. 膨胀污点掩膜一点点，让修补更自然
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(spotMask, spotMask, kernel);

    if (spotCount == 0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 10. 修补
    cv::Mat dst;
    if (options.useInpaint) {
        cv::inpaint(src, spotMask, dst, 3, cv::INPAINT_TELEA);
    } else {
        // 备用方案：中值滤波
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