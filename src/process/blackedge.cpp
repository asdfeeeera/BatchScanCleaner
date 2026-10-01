#include "blackedge.h"

#include <opencv2/imgproc.hpp>

namespace process {

namespace {

double rowMean(const cv::Mat &gray, int y)
{
    if (y < 0 || y >= gray.rows) {
        return 255.0;
    }
    const uchar *row = gray.ptr<uchar>(y);
    long sum = 0;
    for (int x = 0; x < gray.cols; ++x) {
        sum += row[x];
    }
    return static_cast<double>(sum) / gray.cols;
}

double colMean(const cv::Mat &gray, int x)
{
    if (x < 0 || x >= gray.cols) {
        return 255.0;
    }
    long sum = 0;
    for (int y = 0; y < gray.rows; ++y) {
        sum += gray.at<uchar>(y, x);
    }
    return static_cast<double>(sum) / gray.rows;
}

} // namespace

int BlackEdge::detectTop(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    const double threshold = globalMean * options.relativeDarkRatio;
    int count = 0;

    for (int y = 0; y < maxScan && y < gray.rows; ++y) {
        if (rowMean(gray, y) < threshold) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

int BlackEdge::detectBottom(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    const double threshold = globalMean * options.relativeDarkRatio;
    int count = 0;

    for (int i = 0; i < maxScan && i < gray.rows; ++i) {
        const int y = gray.rows - 1 - i;
        if (rowMean(gray, y) < threshold) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

int BlackEdge::detectLeft(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    const double threshold = globalMean * options.relativeDarkRatio;
    int count = 0;

    for (int x = 0; x < maxScan && x < gray.cols; ++x) {
        if (colMean(gray, x) < threshold) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

int BlackEdge::detectRight(const cv::Mat &gray, const BlackEdgeOptions &options, double globalMean)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    const double threshold = globalMean * options.relativeDarkRatio;
    int count = 0;

    for (int i = 0; i < maxScan && i < gray.cols; ++i) {
        const int x = gray.cols - 1 - i;
        if (colMean(gray, x) < threshold) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

BlackEdgeResult BlackEdge::removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options)
{
    BlackEdgeResult result;

    if (src.empty()) {
        return result;
    }

    // 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src;
    }

    // 计算整图平均灰度
    cv::Scalar globalScalar = cv::mean(gray);
    const double globalMean = globalScalar[0];

    // 检测四边
    const int top    = detectTop(gray, options, globalMean);
    const int bottom = detectBottom(gray, options, globalMean);
    const int left   = detectLeft(gray, options, globalMean);
    const int right  = detectRight(gray, options, globalMean);

    if (top == 0 && bottom == 0 && left == 0 && right == 0) {
        result.image = src.clone();
        result.ok = true;
        return result;
    }

    result.topPixels = top;
    result.bottomPixels = bottom;
    result.leftPixels = left;
    result.rightPixels = right;

    if (options.fillWhite) {
        cv::Mat dst = src.clone();
        cv::Scalar white;
        if (dst.channels() == 4) {
            white = cv::Scalar(255, 255, 255, 255);
        } else {
            white = cv::Scalar(255, 255, 255);
        }

        if (top > 0) {
            cv::rectangle(dst, cv::Rect(0, 0, dst.cols, top), white, cv::FILLED);
        }
        if (bottom > 0) {
            cv::rectangle(dst, cv::Rect(0, dst.rows - bottom, dst.cols, bottom),
                          white, cv::FILLED);
        }
        if (left > 0) {
            cv::rectangle(dst, cv::Rect(0, 0, left, dst.rows), white, cv::FILLED);
        }
        if (right > 0) {
            cv::rectangle(dst, cv::Rect(dst.cols - right, 0, right, dst.rows),
                          white, cv::FILLED);
        }

        result.image = dst;
    } else {
        int x0 = (left > 0) ? left : 0;
        int y0 = (top > 0) ? top : 0;
        int x1 = src.cols - ((right > 0) ? right : 0);
        int y1 = src.rows - ((bottom > 0) ? bottom : 0);

        int w = x1 - x0;
        int h = y1 - y0;
        if (w < 1) { w = 1; }
        if (h < 1) { h = 1; }

        cv::Rect roi(x0, y0, w, h);
        roi &= cv::Rect(0, 0, src.cols, src.rows);
        result.image = src(roi).clone();
    }

    result.ok = true;
    return result;
}

} // namespace process