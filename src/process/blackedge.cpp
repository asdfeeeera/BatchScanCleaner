#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace process {

double BlackEdge::estimatePaperGray(const cv::Mat &gray,
                                     const BlackEdgeOptions &options)
{
    cv::Scalar m = cv::mean(gray);
    return m[0];
}

int BlackEdge::scanTop(const cv::Mat &, double, const BlackEdgeOptions &) { return 0; }
int BlackEdge::scanBottom(const cv::Mat &, double, const BlackEdgeOptions &) { return 0; }
int BlackEdge::scanLeft(const cv::Mat &, double, const BlackEdgeOptions &) { return 0; }
int BlackEdge::scanRight(const cv::Mat &, double, const BlackEdgeOptions &) { return 0; }

void BlackEdge::smoothMask(cv::Mat &mask, int kernelSize)
{
    if (kernelSize <= 1) return;
    if (kernelSize % 2 == 0) kernelSize += 1;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

BlackEdgeResult BlackEdge::removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options)
{
    BlackEdgeResult result;
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

    // 2. 降采样加速
    cv::Mat small;
    double scale = 1.0;
    const int maxDim = std::max(W, H);
    if (maxDim > 1500) {
        scale = 1500.0 / maxDim;
        cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        small = gray;
    }

    // 3. 高斯模糊
    cv::GaussianBlur(small, small, cv::Size(5, 5), 0);

    // 4. Otsu 二值化
    cv::Mat binary;
    double otsuThresh = cv::threshold(small, binary, 0, 255,
                                      cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Otsu 阈值过低（整图偏暗），用固定 180 兜底
    if (otsuThresh < 120.0) {
        cv::threshold(small, binary, 180, 255, cv::THRESH_BINARY);
    }

    // 5. 大核闭运算：把文字、表格并入纸张
    //    核大小按图片尺寸自适应
    int closeSize = std::max(31, std::min(small.cols, small.rows) / 40);
    if (closeSize % 2 == 0) closeSize += 1;
    cv::Mat closeKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(closeSize, closeSize));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, closeKernel);

    // 6. 找最大白色连通域
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    if (nLabels <= 1) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    int maxLabel = -1;
    int maxArea = 0;
    for (int i = 1; i < nLabels; ++i) {
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a > maxArea) {
            maxArea = a;
            maxLabel = i;
        }
    }

    if (maxLabel < 0) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 7. 取外接矩形
    int bx = stats.at<int>(maxLabel, cv::CC_STAT_LEFT);
    int by = stats.at<int>(maxLabel, cv::CC_STAT_TOP);
    int bw = stats.at<int>(maxLabel, cv::CC_STAT_WIDTH);
    int bh = stats.at<int>(maxLabel, cv::CC_STAT_HEIGHT);

    // 8. 还原坐标
    if (scale != 1.0) {
        bx = static_cast<int>(bx / scale);
        by = static_cast<int>(by / scale);
        bw = static_cast<int>(bw / scale);
        bh = static_cast<int>(bh / scale);
    }

    // 9. 检查：矩形几乎等于整图 → 认为无黑边
    const int marginX = static_cast<int>(W * 0.005);
    const int marginY = static_cast<int>(H * 0.005);
    const bool touchesLeft = bx <= marginX;
    const bool touchesTop = by <= marginY;
    const bool touchesRight = (bx + bw) >= (W - marginX);
    const bool touchesBottom = (by + bh) >= (H - marginY);

    if (touchesLeft && touchesTop && touchesRight && touchesBottom) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 10. 掩膜：矩形内为白（保留），矩形外为黑（填白）
    cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
    cv::Rect paperRect(bx, by, bw, bh);
    paperRect &= cv::Rect(0, 0, W, H);
    cv::rectangle(mask, paperRect, cv::Scalar(255), cv::FILLED);

    // 11. 掩膜外扩展一点，避免切割纸张边缘
    if (options.expandPixels > 0) {
        const int ksize = options.expandPixels * 2 + 1;
        cv::Mat expandKernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::dilate(mask, mask, expandKernel);
    }

    // 12. 计算填白区域
    cv::Mat invMask;
    cv::bitwise_not(mask, invMask);
    const int nonZeroCount = cv::countNonZero(invMask);

    if (nonZeroCount < static_cast<int>(static_cast<double>(W) * H * 0.001)) {
        result.image = src.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // 13. 填白
    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (dst.channels() == 4) {
        white = cv::Scalar(255, 255, 255, 255);
    } else {
        white = cv::Scalar(255, 255, 255);
    }
    dst.setTo(white, invMask);

    result.image = dst;
    result.filledPixels = nonZeroCount;
    result.topPixels = by;
    result.bottomPixels = H - (by + bh);
    result.leftPixels = bx;
    result.rightPixels = W - (bx + bw);
    result.ok = true;
    return result;
}

} // namespace process