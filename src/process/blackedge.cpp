#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace process {

namespace {

// 统计一行中暗像素比例
double darkRatioInRow(const cv::Mat &gray, int y, int darkThreshold, int x0, int x1)
{
    if (y < 0 || y >= gray.rows) {
        return 0.0;
    }
    const int width = x1 - x0;
    if (width <= 0) {
        return 0.0;
    }

    const uchar *row = gray.ptr<uchar>(y);
    int darkCount = 0;
    for (int x = x0; x < x1; ++x) {
        if (row[x] < darkThreshold) {
            ++darkCount;
        }
    }
    return static_cast<double>(darkCount) / width;
}

// 统计一列中暗像素比例
double darkRatioInCol(const cv::Mat &gray, int x, int darkThreshold, int y0, int y1)
{
    if (x < 0 || x >= gray.cols) {
        return 0.0;
    }
    const int height = y1 - y0;
    if (height <= 0) {
        return 0.0;
    }

    int darkCount = 0;
    for (int y = y0; y < y1; ++y) {
        if (gray.at<uchar>(y, x) < darkThreshold) {
            ++darkCount;
        }
    }
    return static_cast<double>(darkCount) / height;
}

} // namespace

int BlackEdge::detectTop(const cv::Mat &gray, const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    int count = 0;

    for (int y = 0; y < maxScan && y < gray.rows; ++y) {
        const double ratio = darkRatioInRow(gray, y, options.darkThreshold, 0, gray.cols);
        if (ratio >= options.darkRatio) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

int BlackEdge::detectBottom(const cv::Mat &gray, const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.rows * options.maxScanRatio);
    int count = 0;

    for (int i = 0; i < maxScan && i < gray.rows; ++i) {
        const int y = gray.rows - 1 - i;
        const double ratio = darkRatioInRow(gray, y, options.darkThreshold, 0, gray.cols);
        if (ratio >= options.darkRatio) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

int BlackEdge::detectLeft(const cv::Mat &gray, const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    int count = 0;

    for (int x = 0; x < maxScan && x < gray.cols; ++x) {
        const double ratio = darkRatioInCol(gray, x, options.darkThreshold, 0, gray.rows);
        if (ratio >= options.darkRatio) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

int BlackEdge::detectRight(const cv::Mat &gray, const BlackEdgeOptions &options)
{
    const int maxScan = static_cast<int>(gray.cols * options.maxScanRatio);
    int count = 0;

    for (int i = 0; i < maxScan && i < gray.cols; ++i) {
        const int x = gray.cols - 1 - i;
        const double ratio = darkRatioInCol(gray, x, options.darkThreshold, 0, gray.rows);
        if (ratio >= options.darkRatio) {
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

    // 转灰度用于检测
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src;
    }

    // 检测四边黑边宽度
    const int top    = detectTop(gray, options);
    const int bottom = detectBottom(gray, options);
    const int left   = detectLeft(gray, options);
    const int right  = detectRight(gray, options);

    // 四边都没检测到黑边，直接返回原图
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
        // 填白模式：不裁切，保留原图尺寸
        cv::Mat dst = src.clone();

        cv::Scalar white;
        if (dst.channels() == 4) {
            white = cv::Scalar(255, 255, 255, 255);
        } else {
            white = cv::Scalar(255, 255, 255);
        }

        // 上边
        if (top > 0) {
            cv::rectangle(dst,
                          cv::Rect(0, 0, dst.cols, top),
                          white,
                          cv::FILLED);
        }
        // 下边
        if (bottom > 0) {
            cv::rectangle(dst,
                          cv::Rect(0, dst.rows - bottom, dst.cols, bottom),
                          white,
                          cv::FILLED);
        }
        // 左边
        if (left > 0) {
            cv::rectangle(dst,
                          cv::Rect(0, 0, left, dst.rows),
                          white,
                          cv::FILLED);
        }
        // 右边
        if (right > 0) {
            cv::rectangle(dst,
                          cv::Rect(dst.cols - right, 0, right, dst.rows),
                          white,
                          cv::FILLED);
        }

        result.image = dst;
    } else {
        // 安全裁切模式
        const int x0 = (left   > 0) ? left   : 0;
        const int y0 = (top    > 0) ? top    : 0;
        const int x1 = src.cols - ((right  > 0) ? right  : 0);
        const int y1 = src.rows - ((bottom > 0) ? bottom : 0);

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