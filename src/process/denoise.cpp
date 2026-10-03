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
    const double paperGray = estimatePaperGray(gray);
    const double contentThresh = paperGray * 0.85;

    cv::Mat binary;
    cv::threshold(gray, binary, contentThresh, 255, cv::THRESH_BINARY_INV);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    protectMask = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);

    const int protectArea = 5;
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

    if (protectRadius > 0) {
        const int ksize = protectRadius * 2 + 1;
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::dilate(protectMask, protectMask, kernel);
    }
}

namespace {

cv::Mat whitenBackground(const cv::Mat &src, const cv::Mat &protectMask)
{
    if (src.empty()) return src.clone();

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    const int bgSize = 61;
    cv::Mat bgKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(bgSize, bgSize));
    cv::Mat background;
    cv::morphologyEx(gray, background, cv::MORPH_CLOSE, bgKernel);

    cv::Mat diff;
    cv::absdiff(gray, background, diff);

    cv::Mat bgMask;
    cv::threshold(diff, bgMask, 25, 255, cv::THRESH_BINARY_INV);

    cv::bitwise_and(bgMask, ~protectMask, bgMask);

    cv::Mat result = src.clone();
    cv::Scalar white;
    if (result.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else if (result.channels() == 3) {
        white = cv::Scalar(255, 255, 255);
    } else {
        white = cv::Scalar(255);
    }
    result.setTo(white, bgMask);

    return result;
}

} // namespace

DenoiseResult Denoise::removeSpots(const cv::Mat &src,
                                    const DenoiseOptions &options)
{
    DenoiseResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    // 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    // 1. 建保护掩膜
    cv::Mat protectMask;
    buildProtectMask(gray, protectMask, options.protectRadius);

    // 1.1 合并外部保护掩膜
    if (!options.protectMask.empty()) {
        cv::Mat ext = options.protectMask;
        if (ext.size() != gray.size()) {
            cv::Mat tmp;
            cv::resize(ext, tmp, gray.size(), 0, 0, cv::INTER_NEAREST);
            ext = tmp;
        }
        if (ext.type() != CV_8UC1) {
            cv::Mat tmp;
            ext.convertTo(tmp, CV_8UC1);
            ext = tmp;
        }
        cv::bitwise_or(protectMask, ext, protectMask);
    }

    // 2. 填底色
    cv::Mat whitened = whitenBackground(src, protectMask);

    // 3. 在填白后的图上做污点检测
    cv::Mat whitenedGray;
    if (whitened.channels() == 3) {
        cv::cvtColor(whitened, whitenedGray, cv::COLOR_BGR2GRAY);
    } else if (whitened.channels() == 4) {
        cv::cvtColor(whitened, whitenedGray, cv::COLOR_BGRA2GRAY);
    } else {
        whitenedGray = whitened.clone();
    }

    const double paperGray = estimatePaperGray(whitenedGray);
    const double darkThreshold = paperGray * options.darkRatio;

    cv::Mat binary;
    cv::threshold(whitenedGray, binary, darkThreshold, 255, cv::THRESH_BINARY_INV);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    // ★ 普通区域的污点上限（按强度）
    int maxSpotArea = 30;
    if (options.strengthLevel == 0) maxSpotArea = 15;
    else if (options.strengthLevel == 1) maxSpotArea = 30;
    else if (options.strengthLevel == 2) maxSpotArea = 60;

    // ★ 装订孔：边缘区域
    //   竖版（H > W）：左边缘 6% 宽度
    //   横版（W >= H）：上边缘 6% 高度
    const bool isPortrait = (H > W);
    const int edgeThreshold = isPortrait
        ? static_cast<int>(W * 0.06)
        : static_cast<int>(H * 0.06);

    const int bindingMaxArea = 2500;   // 装订孔最大面积

    cv::Mat spotMask = cv::Mat::zeros(H, W, CV_8UC1);
    int spotCount = 0;
    int totalPixels = 0;

    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);

        if (area < 2) continue;

        // ★ 判断该暗块是否落在装订孔区域
        bool isBinding = false;
        if (isPortrait) {
            // 竖版：整条左边缘
            isBinding = ((x + w) < edgeThreshold);
        } else {
            // 横版：整条上边缘
            isBinding = ((y + h) < edgeThreshold);
        }

        const int localMaxArea = isBinding ? bindingMaxArea : maxSpotArea;
        if (area > localMaxArea) continue;

        const double aspect = static_cast<double>(std::max(w, h))
                              / std::max(1, std::min(w, h));
        if (aspect > 5.0) continue;

        cv::Rect r(x, y, w, h);
        r &= cv::Rect(0, 0, W, H);

        // ★ 装订孔区域：即使被 protectMask 保护，也允许去除
        if (!isBinding) {
            cv::Mat roi = protectMask(r);
            if (cv::countNonZero(roi) > 0) continue;
        }

        cv::rectangle(spotMask, r, cv::Scalar(255), cv::FILLED);
        ++spotCount;
        totalPixels += area;
    }

    cv::Mat finalImage;
    if (spotCount == 0) {
        finalImage = whitened;
    } else {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::dilate(spotMask, spotMask, kernel);
        cv::bitwise_and(spotMask, ~protectMask, spotMask);

        cv::Mat dst;
        if (options.useInpaint) {
            cv::inpaint(whitened, spotMask, dst, 3, cv::INPAINT_TELEA);
        } else {
            cv::medianBlur(whitened, dst, 3);
            whitened.copyTo(dst, ~spotMask);
        }
        finalImage = dst;
    }

    result.image = finalImage;
    result.spotCount = spotCount;
    result.cleanedPixels = totalPixels;
    result.ok = true;
    return result;
}

} // namespace process