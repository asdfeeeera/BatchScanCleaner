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
    // 先估算纸张灰度
    const double paperGray = estimatePaperGray(gray);

    // 分界线：纸张 × 0.75
    const double contentThreshold = paperGray * 0.75;

    // 统计所有低于阈值的像素
    long sum = 0;
    int count = 0;
    const int W = gray.cols;
    const int H = gray.rows;
    for (int y = 0; y < H; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = 0; x < W; x += 2) {
            const uchar v = row[x];
            if (v < contentThreshold) {
                sum += v;
                ++count;
            }
        }
    }

    // 像素太少：说明整张图很淡，用虚拟暗部灰度
    if (count < 100) {
        return paperGray * 0.5;
    }

    return static_cast<double>(sum) / count;
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

    // 纸张太暗，跳过
    if (paperGray < 150.0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 强度 → gamma
    double gamma = 1.0;
    if (options.strengthLevel == 0) gamma = 1.3;
    else if (options.strengthLevel == 1) gamma = 1.6;
    else if (options.strengthLevel == 2) gamma = 2.0;

    // LUT 映射
    // 关键改动：分界线用 paperGray * 0.75，避免整图被加深
    const double contentThreshold = paperGray * 0.75;

    uchar lut[256];
    for (int v = 0; v < 256; ++v) {
        if (v <= darkGray) {
            // 暗部 → 拉到纯黑
            lut[v] = 0;
        } else if (v >= paperGray) {
            // 亮部 → 拉到纯白
            lut[v] = 255;
        } else if (v < contentThreshold) {
            // 文字候选区 → 强拉伸
            double t = (v - darkGray) / std::max(1.0, contentThreshold - darkGray);
            t = std::pow(t, gamma);
            lut[v] = static_cast<uchar>(std::min(255.0, t * 128.0));
        } else {
            // 背景区 → 温和拉伸到白
            double t = (v - contentThreshold) / std::max(1.0, paperGray - contentThreshold);
            lut[v] = static_cast<uchar>(std::min(255.0, 128.0 + t * 127.0));
        }
    }

    cv::Mat lutMat(1, 256, CV_8UC1, lut);

    // 应用
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

        cv::Mat lab;
        cv::cvtColor(srcBgr, lab, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> labCh;
        cv::split(lab, labCh);

        cv::Mat LEnhanced;
        cv::LUT(labCh[0], lutMat, LEnhanced);

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