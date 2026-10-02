#include "enhance.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace process {

double Enhance::estimatePaperGray(const cv::Mat &gray)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int mx = static_cast<int>(W * 0.20);
    const int my = static_cast<int>(H * 0.20);

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

double Enhance::estimateDarkGray(const cv::Mat &gray)
{
    const int W = gray.cols;
    const int H = gray.rows;

    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(W) * H / 4);
    for (int y = 0; y < H; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = 0; x < W; x += 2) {
            pixels.push_back(row[x]);
        }
    }
    if (pixels.empty()) return 0.0;
    const size_t idx = pixels.size() * 5 / 100;
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

void Enhance::buildColorProtectMask(const cv::Mat &src,
                                     cv::Mat &colorMask,
                                     int saturationThreshold)
{
    colorMask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    if (src.channels() < 3) return;

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat &sat = channels[1];

    cv::threshold(sat, colorMask, saturationThreshold, 255, cv::THRESH_BINARY);
}

EnhanceResult Enhance::enhanceText(const cv::Mat &src,
                                    const EnhanceOptions &options)
{
    EnhanceResult result;
    if (src.empty()) return result;

    // 1. 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    const double paperGray = estimatePaperGray(gray);
    const double darkGray = estimateDarkGray(gray);

    result.paperGray = paperGray;
    result.darkGray = darkGray;

    // 纸张本身太暗，跳过
    if (paperGray < 150.0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 2. 强度对应 gamma
    double gamma = 1.0;
    if (options.strengthLevel == 0) gamma = 1.2;
    else if (options.strengthLevel == 1) gamma = 1.5;
    else if (options.strengthLevel == 2) gamma = 2.0;

    // 3. 构建 LUT（映射曲线）
    uchar lut[256];
    const double paperTarget = std::min(255.0, paperGray * 1.05);
    const double darkTarget = static_cast<double>(options.targetDarkGray);

    for (int v = 0; v < 256; ++v) {
        if (v <= darkGray) {
            lut[v] = static_cast<uchar>(darkTarget);
        } else if (v >= paperGray) {
            lut[v] = 255;
        } else {
            double t = (v - darkGray) / (paperGray - darkGray);
            t = std::pow(t, gamma);
            lut[v] = static_cast<uchar>(std::min(255.0, t * 255.0));
        }
    }

    cv::Mat lutMat(1, 256, CV_8UC1, lut);

    // 4. 应用
    cv::Mat output;
    if (src.channels() == 1) {
        cv::LUT(src, lutMat, output);
        result.enhancedPixels = static_cast<int>(src.total());
    } else {
        cv::Mat srcBgr;
        if (src.channels() == 4) {
            cv::cvtColor(src, srcBgr, cv::COLOR_BGRA2BGR);
        } else {
            srcBgr = src;
        }

        // 用 Lab 空间，只改 L 通道（亮度），保留色相
        cv::Mat lab;
        cv::cvtColor(srcBgr, lab, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> labCh;
        cv::split(lab, labCh);

        cv::Mat LEnhanced;
        cv::LUT(labCh[0], lutMat, LEnhanced);

        // 彩色保护：彩色像素保留原始 L
        if (options.protectColor) {
            cv::Mat colorMask;
            buildColorProtectMask(srcBgr, colorMask, options.colorSaturationThreshold);
            LEnhanced.copyTo(labCh[0], colorMask);
            result.enhancedPixels =
                static_cast<int>(srcBgr.total()) - cv::countNonZero(colorMask);
        } else {
            result.enhancedPixels = static_cast<int>(srcBgr.total());
        }

        cv::merge(labCh, lab);
        cv::cvtColor(lab, output, cv::COLOR_Lab2BGR);
    }

    result.image = output;
    result.ok = true;
    return result;
}

} // namespace process