#include "colorline.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace process {

void ColorLine::buildColorMask(const cv::Mat &bgr,
                                cv::Mat &colorMask,
                                int satThreshold,
                                int valThreshold)
{
    colorMask = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8UC1);
    if (bgr.channels() < 3) return;

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    cv::Mat &S = ch[1];
    cv::Mat &V = ch[2];

    // 通道 1：高饱和彩色
    cv::Mat mask1, mask2;
    cv::threshold(S, mask1, satThreshold, 255, cv::THRESH_BINARY);
    cv::threshold(V, mask2, valThreshold, 255, cv::THRESH_BINARY);
    cv::bitwise_and(mask1, mask2, colorMask);

    // 通道 2：弱彩色——通道间差异检测（捕捉淡蓝、淡绿、淡红）
    std::vector<cv::Mat> bgrCh;
    cv::split(bgr, bgrCh);
    cv::Mat &B = bgrCh[0];
    cv::Mat &G = bgrCh[1];
    cv::Mat &R = bgrCh[2];

    // 淡蓝：B - max(R,G) > 15
    cv::Mat rgMax;
    cv::max(R, G, rgMax);
    cv::Mat bDiff;
    cv::subtract(B, rgMax, bDiff);
    cv::Mat bWeak;
    cv::threshold(bDiff, bWeak, 15, 255, cv::THRESH_BINARY);

    // 淡绿：G - max(R,B) > 15
    cv::Mat rbMax;
    cv::max(R, B, rbMax);
    cv::Mat gDiff;
    cv::subtract(G, rbMax, gDiff);
    cv::Mat gWeak;
    cv::threshold(gDiff, gWeak, 15, 255, cv::THRESH_BINARY);

    // 淡红：R - max(G,B) > 15
    cv::Mat gbMax;
    cv::max(G, B, gbMax);
    cv::Mat rDiff;
    cv::subtract(R, gbMax, rDiff);
    cv::Mat rWeak;
    cv::threshold(rDiff, rWeak, 15, 255, cv::THRESH_BINARY);

    // 合并弱彩色
    cv::Mat weakColor;
    cv::bitwise_or(bWeak, gWeak, weakColor);
    cv::bitwise_or(weakColor, rWeak, weakColor);

    // 排除暗色（避免黑边被当彩色）
    cv::Mat bright;
    cv::threshold(V, bright, 80, 255, cv::THRESH_BINARY);
    cv::bitwise_and(weakColor, bright, weakColor);

    // 合并两个通道
    cv::bitwise_or(colorMask, weakColor, colorMask);
}

namespace {

cv::Scalar averageColor(const cv::Mat &bgr, const cv::Mat &mask, int label)
{
    long b = 0, g = 0, r = 0;
    int count = 0;
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b *bgrRow = bgr.ptr<cv::Vec3b>(y);
        const int *labelRow = mask.ptr<int>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            if (labelRow[x] == label) {
                b += bgrRow[x][0];
                g += bgrRow[x][1];
                r += bgrRow[x][2];
                ++count;
            }
        }
    }
    if (count == 0) return cv::Scalar(0, 0, 0);
    return cv::Scalar(b / count, g / count, r / count);
}

} // namespace

ColorLineResult ColorLine::detect(const cv::Mat &src,
                                   const ColorLineOptions &options)
{
    ColorLineResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    cv::Mat bgr;
    if (src.channels() == 1) {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    } else {
        bgr = src.clone();
    }

    cv::Mat colorMask;
    buildColorMask(bgr, colorMask,
                   options.saturationThreshold,
                   options.valueThreshold);

    if (options.edgeMarginRatio > 0) {
        const int mx = static_cast<int>(W * options.edgeMarginRatio);
        const int my = static_cast<int>(H * options.edgeMarginRatio);
        cv::rectangle(colorMask, cv::Rect(0, 0, W, my), cv::Scalar(0), cv::FILLED);
        cv::rectangle(colorMask, cv::Rect(0, H - my, W, my), cv::Scalar(0), cv::FILLED);
        cv::rectangle(colorMask, cv::Rect(0, 0, mx, H), cv::Scalar(0), cv::FILLED);
        cv::rectangle(colorMask, cv::Rect(W - mx, 0, mx, H), cv::Scalar(0), cv::FILLED);
    }

    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(colorMask, dilated, kernel);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        dilated, labels, stats, centroids, 8, CV_32S);

    for (int i = 1; i < nLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);

        if (area < 20) continue;
        if (area > options.maxArea) continue;

        const int longSide = std::max(w, h);
        const int shortSide = std::max(1, std::min(w, h));
        const double aspect = static_cast<double>(longSide) / shortSide;

        if (longSide < options.minLength) continue;
        if (aspect < options.minAspectRatio) continue;

        ColorLineItem item;
        item.boundingBox = cv::Rect(x, y, w, h);
        item.length = longSide;
        item.aspectRatio = aspect;
        item.area = area;
        item.color = averageColor(bgr, labels, i);

        result.items.push_back(item);
    }

    cv::Mat marked = bgr.clone();
    for (const auto &item : result.items) {
        cv::Rect r = item.boundingBox;
        r.x -= 3;
        r.y -= 3;
        r.width += 6;
        r.height += 6;
        r &= cv::Rect(0, 0, W, H);
        cv::rectangle(marked, r, cv::Scalar(0, 255, 255), 3);
    }

    result.image = bgr;
    result.markedImage = marked;
    result.ok = true;

    if (result.items.empty()) {
        result.skipped = true;
    }
    return result;
}

ColorLineResult ColorLine::clear(const cv::Mat &src,
                                  const ColorLineOptions &options)
{
    ColorLineResult result = detect(src, options);

    if (!result.ok || result.items.empty()) {
        return result;
    }

    cv::Mat bgr = result.image.clone();
    const int W = bgr.cols;
    const int H = bgr.rows;

    cv::Mat clearMask = cv::Mat::zeros(H, W, CV_8UC1);
    for (const auto &item : result.items) {
        cv::Rect r = item.boundingBox;
        r.x -= 2;
        r.y -= 2;
        r.width += 4;
        r.height += 4;
        r &= cv::Rect(0, 0, W, H);
        cv::rectangle(clearMask, r, cv::Scalar(255), cv::FILLED);
    }

    cv::Mat colorMask;
    buildColorMask(bgr, colorMask,
                   options.saturationThreshold,
                   options.valueThreshold);
    cv::bitwise_and(clearMask, colorMask, clearMask);

    cv::Mat dilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(clearMask, clearMask, dilateKernel);

    cv::Mat dst = bgr.clone();
    dst.setTo(cv::Scalar(255, 255, 255), clearMask);

    result.image = dst;
    result.clearedPixels = cv::countNonZero(clearMask);
    return result;
}

} // namespace process