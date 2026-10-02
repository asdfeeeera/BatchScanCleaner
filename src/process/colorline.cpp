#include "colorline.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace process {

void ColorLine::buildColorMask(const cv::Mat &bgr,
                                cv::Mat &colorMask,
                                const ColorLineOptions &options)
{
    colorMask = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8UC1);
    if (bgr.channels() < 3) return;

    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);
    cv::Mat &B = ch[0];
    cv::Mat &G = ch[1];
    cv::Mat &R = ch[2];

    cv::Mat maxRG, maxRGB, minRG, minRGB;
    cv::max(R, G, maxRG);
    cv::max(maxRG, B, maxRGB);
    cv::min(R, G, minRG);
    cv::min(minRG, B, minRGB);

    cv::Mat colorfulness;
    cv::subtract(maxRGB, minRGB, colorfulness);

    cv::Mat brightness;
    cv::add(B, G, brightness);
    cv::add(brightness, R, brightness);
    cv::divide(brightness, 3.0, brightness);

    cv::Mat c1, c2;
    cv::threshold(colorfulness, c1, options.channelDiffThreshold, 255, cv::THRESH_BINARY);
    cv::threshold(brightness, c2, options.valueThreshold, 255, cv::THRESH_BINARY);
    cv::bitwise_and(c1, c2, colorMask);
}

namespace {

// 计算矩阵某区域的中位数
double medianOf(std::vector<double> &v)
{
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// 计算一行 y 的色偏（B - max(G,R) 的中位数 等），返回三通道偏差最大值
double rowColorBias(const cv::Mat &B, const cv::Mat &G, const cv::Mat &R,
                    int y, int x0, int x1)
{
    std::vector<double> bBias, gBias, rBias;
    bBias.reserve(x1 - x0);
    gBias.reserve(x1 - x0);
    rBias.reserve(x1 - x0);

    const uchar *bRow = B.ptr<uchar>(y);
    const uchar *gRow = G.ptr<uchar>(y);
    const uchar *rRow = R.ptr<uchar>(y);

    for (int x = x0; x < x1; ++x) {
        const double b = bRow[x];
        const double g = gRow[x];
        const double r = rRow[x];

        bBias.push_back(b - std::max(g, r));
        gBias.push_back(g - std::max(b, r));
        rBias.push_back(r - std::max(b, g));
    }

    const double mb = medianOf(bBias);
    const double mg = medianOf(gBias);
    const double mr = medianOf(rBias);

    return std::max({mb, mg, mr});
}

// 计算一列 x 的色偏
double colColorBias(const cv::Mat &B, const cv::Mat &G, const cv::Mat &R,
                    int x, int y0, int y1)
{
    std::vector<double> bBias, gBias, rBias;
    bBias.reserve(y1 - y0);
    gBias.reserve(y1 - y0);
    rBias.reserve(y1 - y0);

    for (int y = y0; y < y1; ++y) {
        const double b = B.at<uchar>(y, x);
        const double g = G.at<uchar>(y, x);
        const double r = R.at<uchar>(y, x);

        bBias.push_back(b - std::max(g, r));
        gBias.push_back(g - std::max(b, r));
        rBias.push_back(r - std::max(b, g));
    }

    const double mb = medianOf(bBias);
    const double mg = medianOf(gBias);
    const double mr = medianOf(rBias);

    return std::max({mb, mg, mr});
}

cv::Scalar avgRowColor(const cv::Mat &bgr, int y, int x0, int x1)
{
    long b = 0, g = 0, r = 0;
    const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
    for (int x = x0; x < x1; ++x) {
        b += row[x][0];
        g += row[x][1];
        r += row[x][2];
    }
    const int n = x1 - x0;
    if (n <= 0) return cv::Scalar(0, 0, 0);
    return cv::Scalar(b / n, g / n, r / n);
}

cv::Scalar avgColColor(const cv::Mat &bgr, int x, int y0, int y1)
{
    long b = 0, g = 0, r = 0;
    for (int y = y0; y < y1; ++y) {
        const cv::Vec3b &px = bgr.at<cv::Vec3b>(y, x);
        b += px[0];
        g += px[1];
        r += px[2];
    }
    const int n = y1 - y0;
    if (n <= 0) return cv::Scalar(0, 0, 0);
    return cv::Scalar(b / n, g / n, r / n);
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

    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);

    const int mx = static_cast<int>(W * options.edgeMarginRatio);
    const int my = static_cast<int>(H * options.edgeMarginRatio);
    const int x0 = mx;
    const int x1 = W - mx;
    const int y0 = my;
    const int y1 = H - my;

    if (x1 <= x0 || y1 <= y0) {
        result.image = bgr;
        result.markedImage = bgr.clone();
        result.ok = true;
        result.skipped = true;
        return result;
    }

    // ---- 逐行色偏分析 ----
    std::vector<double> rowBias(H, 0.0);
    for (int y = y0; y < y1; ++y) {
        rowBias[y] = rowColorBias(ch[0], ch[1], ch[2], y, x0, x1);
    }

    // 计算行色偏的统计值
    std::vector<double> validRowBias(rowBias.begin() + y0, rowBias.begin() + y1);
    std::vector<double> sortedRow = validRowBias;
    std::sort(sortedRow.begin(), sortedRow.end());
    const double rowMedian = sortedRow[sortedRow.size() / 2];

    double rowVariance = 0.0;
    for (double v : validRowBias) rowVariance += (v - rowMedian) * (v - rowMedian);
    rowVariance /= std::max<size_t>(1, validRowBias.size());
    const double rowStd = std::sqrt(rowVariance);

    // 阈值：中位数 + max(3, 3×标准差)
    const double rowThreshold = rowMedian + std::max(3.0, 3.0 * rowStd);

    // 找出超阈值的行
    std::vector<bool> rowIsLine(H, false);
    for (int y = y0; y < y1; ++y) {
        if (rowBias[y] > rowThreshold && rowBias[y] > 2.0) {
            rowIsLine[y] = true;
        }
    }

    // 合并相邻行
    int runStart = -1;
    for (int y = y0; y <= y1; ++y) {
        const bool isLine = (y < y1) && rowIsLine[y];
        if (isLine && runStart < 0) {
            runStart = y;
        } else if (!isLine && runStart >= 0) {
            const int thickness = y - runStart;
            if (thickness <= options.maxThickness) {
                ColorLineItem item;
                item.horizontal = true;
                item.position = runStart;
                item.thickness = thickness;
                item.colorRatio = rowBias[runStart];
                item.color = avgRowColor(bgr, runStart + thickness / 2, x0, x1);
                result.items.push_back(item);
            }
            runStart = -1;
        }
    }

    // ---- 逐列色偏分析 ----
    std::vector<double> colBias(W, 0.0);
    for (int x = x0; x < x1; ++x) {
        colBias[x] = colColorBias(ch[0], ch[1], ch[2], x, y0, y1);
    }

    std::vector<double> validColBias(colBias.begin() + x0, colBias.begin() + x1);
    std::vector<double> sortedCol = validColBias;
    std::sort(sortedCol.begin(), sortedCol.end());
    const double colMedian = sortedCol[sortedCol.size() / 2];

    double colVariance = 0.0;
    for (double v : validColBias) colVariance += (v - colMedian) * (v - colMedian);
    colVariance /= std::max<size_t>(1, validColBias.size());
    const double colStd = std::sqrt(colVariance);

    const double colThreshold = colMedian + std::max(3.0, 3.0 * colStd);

    std::vector<bool> colIsLine(W, false);
    for (int x = x0; x < x1; ++x) {
        if (colBias[x] > colThreshold && colBias[x] > 2.0) {
            colIsLine[x] = true;
        }
    }

    runStart = -1;
    for (int x = x0; x <= x1; ++x) {
        const bool isLine = (x < x1) && colIsLine[x];
        if (isLine && runStart < 0) {
            runStart = x;
        } else if (!isLine && runStart >= 0) {
            const int thickness = x - runStart;
            if (thickness <= options.maxThickness) {
                ColorLineItem item;
                item.horizontal = false;
                item.position = runStart;
                item.thickness = thickness;
                item.colorRatio = colBias[runStart];
                item.color = avgColColor(bgr, runStart + thickness / 2, y0, y1);
                result.items.push_back(item);
            }
            runStart = -1;
        }
    }

    // ---- 标记 ----
    cv::Mat marked = bgr.clone();
    for (const auto &item : result.items) {
        if (item.horizontal) {
            cv::Rect r(0, item.position, W, item.thickness);
            cv::rectangle(marked, r, cv::Scalar(0, 255, 255), 2);
        } else {
            cv::Rect r(item.position, 0, item.thickness, H);
            cv::rectangle(marked, r, cv::Scalar(0, 255, 255), 2);
        }
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
        if (item.horizontal) {
            const int y = std::max(0, item.position - 2);
            const int h = std::min(H - y, item.thickness + 4);
            cv::rectangle(clearMask, cv::Rect(0, y, W, h),
                          cv::Scalar(255), cv::FILLED);
        } else {
            const int x = std::max(0, item.position - 2);
            const int w = std::min(W - x, item.thickness + 4);
            cv::rectangle(clearMask, cv::Rect(x, 0, w, H),
                          cv::Scalar(255), cv::FILLED);
        }
    }

    // 只清除"像素级彩色"的位置（避免全行刷白）
    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);
    cv::Mat maxRG, maxRGB, minRG, minRGB;
    cv::max(ch[2], ch[1], maxRG);
    cv::max(maxRG, ch[0], maxRGB);
    cv::min(ch[2], ch[1], minRG);
    cv::min(minRG, ch[0], minRGB);

    cv::Mat colorfulness;
    cv::subtract(maxRGB, minRGB, colorfulness);

    // 像素级色偏 >= 2 就算彩色
    cv::Mat pixelColor;
    cv::threshold(colorfulness, pixelColor, 2, 255, cv::THRESH_BINARY);

    cv::bitwise_and(clearMask, pixelColor, clearMask);

    // 膨胀让边缘平滑
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