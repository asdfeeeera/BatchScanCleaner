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
    const double paperGray = estimatePaperGray(gray);
    const double contentThreshold = paperGray * 0.75;

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

    cv::Mat srcBgr;
    if (src.channels() == 4) {
        cv::cvtColor(src, srcBgr, cv::COLOR_BGRA2BGR);
    } else {
        srcBgr = src;
    }

    // 1. HSV 饱和度
    cv::Mat hsv;
    cv::cvtColor(srcBgr, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvCh;
    cv::split(hsv, hsvCh);
    cv::Mat hsvColorMask;
    cv::threshold(hsvCh[1], hsvColorMask, saturationThreshold, 255,
                  cv::THRESH_BINARY);

    // 2. Lab a/b 偏离 128
    cv::Mat lab;
    cv::cvtColor(srcBgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labCh;
    cv::split(lab, labCh);

    cv::Mat aDiff;
    cv::absdiff(labCh[1], cv::Scalar(128), aDiff);
    cv::Mat bDiff;
    cv::absdiff(labCh[2], cv::Scalar(128), bDiff);

    cv::Mat aMask;
    cv::threshold(aDiff, aMask, 6, 255, cv::THRESH_BINARY);
    cv::Mat bMask;
    cv::threshold(bDiff, bMask, 6, 255, cv::THRESH_BINARY);

    cv::Mat labColorMask;
    cv::bitwise_or(aMask, bMask, labColorMask);

    // 3. 并集
    cv::bitwise_or(hsvColorMask, labColorMask, colorMask);
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

    // ---- 跳过条件 1：纸张太暗 ----
    if (paperGray < 150.0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // ---- 跳过条件 2：文字已足够黑 ----
    // 文字灰度 <= 阈值（默认 110）→ 不需要加深
    if (darkGray <= static_cast<double>(options.minDarkGrayToEnhance)) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 从 options 读取参数
    double gamma = options.gamma;
    if (gamma < 0.5) gamma = 0.5;
    if (gamma > 4.0) gamma = 4.0;

    int dTarget = options.targetDarkGray;
    if (dTarget < 0) dTarget = 0;
    if (dTarget > 100) dTarget = 100;

    int pTarget = options.targetPaperGray;
    if (pTarget < 200) pTarget = 200;
    if (pTarget > 255) pTarget = 255;

    const double midPoint = (dTarget + pTarget) * 0.5;
    const double contentThreshold = paperGray * 0.75;

    uchar lut[256];
    for (int v = 0; v < 256; ++v) {
        if (v <= darkGray) {
            lut[v] = static_cast<uchar>(dTarget);
        } else if (v >= paperGray) {
            lut[v] = static_cast<uchar>(pTarget);
        } else if (v < contentThreshold) {
            double t = (v - darkGray) /
                       std::max(1.0, contentThreshold - darkGray);
            t = std::pow(t, gamma);
            lut[v] = static_cast<uchar>(dTarget + t * (midPoint - dTarget));
        } else {
            double t = (v - contentThreshold) /
                       std::max(1.0, paperGray - contentThreshold);
            lut[v] = static_cast<uchar>(midPoint + t * (pTarget - midPoint));
        }
    }

    cv::Mat lutMat(1, 256, CV_8UC1, lut);

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
            buildColorProtectMask(srcBgr, colorMask,
                                   options.colorSaturationThreshold);

            cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(5, 5));
            cv::Mat colorMaskDilated;
            cv::dilate(colorMask, colorMaskDilated, k);

            labCh[0].copyTo(LEnhanced, colorMaskDilated);

            result.enhancedPixels =
                static_cast<int>(srcBgr.total()) -
                cv::countNonZero(colorMaskDilated);
        } else {
            result.enhancedPixels = static_cast<int>(srcBgr.total());
        }

        LEnhanced.copyTo(labCh[0]);

        cv::merge(labCh, lab);
        cv::cvtColor(lab, output, cv::COLOR_Lab2BGR);
    }

    result.image = output;
    result.ok = true;
    return result;
}

} // namespace process