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

void Denoise::detectYellowBlobs(const cv::Mat &src,
                                 std::vector<cv::Rect> &outBlobs)
{
    outBlobs.clear();
    if (src.empty() || src.channels() < 3) return;

    cv::Mat bgr;
    if (src.channels() == 4) {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    } else {
        bgr = src;
    }

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvCh;
    cv::split(hsv, hsvCh);

    cv::Mat hMask, sMask, vMask, mask;
    cv::inRange(hsvCh[0], 15, 45, hMask);
    cv::inRange(hsvCh[1], 50, 255, sMask);
    cv::inRange(hsvCh[2], 60, 255, vMask);

    cv::bitwise_and(hMask, sMask, mask);
    cv::bitwise_and(mask, vMask, mask);

    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(
        mask, labels, stats, centroids, 8, CV_32S);

    const int W = src.cols;
    const int H = src.rows;

    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < 100) continue;

        cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                   stats.at<int>(i, cv::CC_STAT_TOP),
                   stats.at<int>(i, cv::CC_STAT_WIDTH),
                   stats.at<int>(i, cv::CC_STAT_HEIGHT));
        r &= cv::Rect(0, 0, W, H);

        if (r.width <= 0 || r.height <= 0) continue;

        outBlobs.push_back(r);
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

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    cv::Mat protectMask;
    buildProtectMask(gray, protectMask, options.protectRadius);

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

    cv::Mat whitened = whitenBackground(src, protectMask);

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
    cv::Mat bindingMask = cv::Mat::zeros(H, W, CV_8UC1);

    int spotCount = 0;
    int totalPixels = 0;

    const bool isPortrait = (H > W);
    const int edgeThreshold = isPortrait
        ? static_cast<int>(W * 0.06)
        : static_cast<int>(H * 0.06);

    // ★ 放宽装订孔面积范围
    const int bindingMinArea = 80;     // 80~10000
    const int bindingMaxArea = 10000;

    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);

        if (area < 2) continue;

        cv::Rect r(x, y, w, h);
        r &= cv::Rect(0, 0, W, H);
        if (r.width <= 0 || r.height <= 0) continue;

        // ★ 装订孔：位置在边缘 + 面积符合 + 不含太多"白"像素
        bool isBinding = false;
        if (isPortrait) {
            isBinding = (x < edgeThreshold);
        } else {
            isBinding = (y < edgeThreshold);
        }

        if (isBinding
            && area >= bindingMinArea
            && area <= bindingMaxArea) {
            // 该区域在原始灰度图上的平均灰度
            cv::Mat roiGray = gray(r);
            const double meanVal = cv::mean(roiGray)[0];
            // ★ 从 100 放宽到 130
            if (meanVal < 130.0) {
                cv::rectangle(bindingMask, r, cv::Scalar(255), cv::FILLED);
                ++spotCount;
                totalPixels += area;
                continue;
            }
        }

        // 普通小污点
        if (area <= maxSpotArea) {
            cv::Mat roiP = protectMask(r);
            if (cv::countNonZero(roiP) > 0) continue;
            cv::rectangle(spotMask, r, cv::Scalar(255), cv::FILLED);
            ++spotCount;
            totalPixels += area;
        }
    }

    cv::Mat finalImage;
    if (spotCount == 0) {
        finalImage = whitened;
    } else {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

        cv::dilate(spotMask, spotMask, kernel);
        cv::bitwise_and(spotMask, ~protectMask, spotMask);

        cv::dilate(bindingMask, bindingMask, kernel);

        cv::Mat finalMask;
        cv::bitwise_or(spotMask, bindingMask, finalMask);

        cv::Mat dst;
        if (options.useInpaint) {
            cv::inpaint(whitened, finalMask, dst, 3, cv::INPAINT_TELEA);
        } else {
            cv::medianBlur(whitened, dst, 3);
            whitened.copyTo(dst, ~finalMask);
        }
        finalImage = dst;
    }

    detectYellowBlobs(src, result.yellowBlobs);

    result.image = finalImage;
    result.spotCount = spotCount;
    result.cleanedPixels = totalPixels;
    result.ok = true;
    return result;
}

} // namespace process