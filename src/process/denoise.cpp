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
    // 用相对纸张灰度检测内容（包括淡文字）
    const double paperGray = estimatePaperGray(gray);
    const double contentThresh = paperGray * 0.85;

    cv::Mat binary;
    cv::threshold(gray, binary, contentThresh, 255, cv::THRESH_BINARY_INV);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    protectMask = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);

    // 保护所有面积 >= 5 的暗块
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

// 填底色：只把"与局部背景差距小"的像素填白，不碰文字
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

    // 大核闭运算估计局部背景（核越大，背景估计越平滑）
    const int bgSize = 61;
    cv::Mat bgKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(bgSize, bgSize));
    cv::Mat background;
    cv::morphologyEx(gray, background, cv::MORPH_CLOSE, bgKernel);

    // 原图与背景的差距
    cv::Mat diff;
    cv::absdiff(gray, background, diff);

    // 差距 < 25 的像素认为"是背景"
    cv::Mat bgMask;
    cv::threshold(diff, bgMask, 25, 255, cv::THRESH_BINARY_INV);

    // 保护掩膜内的像素（文字区域）不算背景
    cv::bitwise_and(bgMask, ~protectMask, bgMask);

    // 只对背景填白
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

    // 1. 先建保护掩膜（文字、线条、印章、签名）
    cv::Mat protectMask;
    buildProtectMask(gray, protectMask, options.protectRadius);

    // 1.1 ★ 合并外部传入的保护掩膜（签名印章保护）
    if (!options.protectMask.empty()) {
        cv::Mat ext = options.protectMask;
        // 尺寸对齐
        if (ext.size() != gray.size()) {
            cv::Mat tmp;
            cv::resize(ext, tmp, gray.size(), 0, 0, cv::INTER_NEAREST);
            ext = tmp;
        }
        // 类型对齐
        if (ext.type() != CV_8UC1) {
            cv::Mat tmp;
            ext.convertTo(tmp, CV_8UC1);
            ext = tmp;
        }
        cv::bitwise_or(protectMask, ext, protectMask);
    }

    // 2. 填底色（保护掩膜内的像素不动）
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

        if (area < 2) continue;
        if (area > maxSpotArea) continue;

        const double aspect = static_cast<double>(std::max(w, h))
                              / std::max(1, std::min(w, h));
        if (aspect > 5.0) continue;

        cv::Rect r(x, y, w, h);
        r &= cv::Rect(0, 0, W, H);

        cv::Mat roi = protectMask(r);
        if (cv::countNonZero(roi) > 0) continue;

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