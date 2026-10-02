#include "stamp_protect.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace protect {

namespace {

// 保证 src 是 BGR 3 通道
cv::Mat toBgr(const cv::Mat &src)
{
    cv::Mat bgr;
    if (src.channels() == 3) {
        bgr = src;
    } else if (src.channels() == 4) {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    } else if (src.channels() == 1) {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = src;
    }
    return bgr;
}

} // namespace

// ============================================================
// 主检测函数
// ============================================================
StampProtectResult StampProtect::detect(const cv::Mat &src,
                                         const StampProtectOptions &options)
{
    StampProtectResult result;
    if (src.empty()) return result;

    const cv::Mat bgr = toBgr(src);
    const int W = bgr.cols;
    const int H = bgr.rows;

    // 转 HSV（H: 0-180, S: 0-255, V: 0-255）
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvCh;
    cv::split(hsv, hsvCh);
    const cv::Mat &Hc = hsvCh[0];
    const cv::Mat &Sc = hsvCh[1];
    const cv::Mat &Vc = hsvCh[2];

    // 总彩色掩膜
    cv::Mat colorMask = cv::Mat::zeros(H, W, CV_8UC1);

    // ---------- 1. 红色（两段） ----------
    if (options.enableRed) {
        cv::Mat red1, red2, redSat, redVal, redMask;
        cv::inRange(Hc, options.redHueLow1, options.redHueHigh1, red1);
        cv::inRange(Hc, options.redHueLow2, options.redHueHigh2, red2);
        cv::bitwise_or(red1, red2, redMask);

        cv::inRange(Sc, options.redSatMin, 255, redSat);
        cv::inRange(Vc, options.redValMin, 255, redVal);
        cv::bitwise_and(redMask, redSat, redMask);
        cv::bitwise_and(redMask, redVal, redMask);
        cv::bitwise_or(colorMask, redMask, colorMask);
    }

    // ---------- 2. 蓝色 ----------
    if (options.enableBlue) {
        cv::Mat blue, blueSat, blueVal, blueMask;
        cv::inRange(Hc, options.blueHueLow, options.blueHueHigh, blueMask);
        cv::inRange(Sc, options.blueSatMin, 255, blueSat);
        cv::inRange(Vc, options.blueValMin, 255, blueVal);
        cv::bitwise_and(blueMask, blueSat, blueMask);
        cv::bitwise_and(blueMask, blueVal, blueMask);
        cv::bitwise_or(colorMask, blueMask, colorMask);
    }

    // ---------- 3. 其它彩色（高饱和 + 非红非蓝） ----------
    if (options.enableOtherColor) {
        cv::Mat sat, val, otherMask;
        cv::inRange(Sc, options.colorSatMin, 255, sat);
        cv::inRange(Vc, options.colorValMin, 255, val);
        cv::bitwise_and(sat, val, otherMask);
        cv::bitwise_or(colorMask, otherMask, colorMask);
    }

    // ---------- 4. 黑色印章（实心黑块） ----------
    cv::Mat blackMask = cv::Mat::zeros(H, W, CV_8UC1);
    if (options.protectBlackStamp) {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        cv::Mat blackBin;
        cv::threshold(gray, blackBin, options.blackValMax, 255,
                      cv::THRESH_BINARY_INV);

        // 开运算去掉细小笔画，只保留粗块
        cv::Mat openKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(9, 9));
        cv::Mat opened;
        cv::morphologyEx(blackBin, opened, cv::MORPH_OPEN, openKernel);

        // 连通域分析，保留大且实心的块
        cv::Mat labels, stats, centroids;
        const int n = cv::connectedComponentsWithStats(
            opened, labels, stats, centroids, 8, CV_32S);

        for (int i = 1; i < n; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            if (area < options.blackMinArea) continue;

            const double fill = static_cast<double>(area) /
                                std::max(1, w * h);
            if (fill < options.blackFillRatio) continue;

            // 把这个连通域画到 blackMask 上
            cv::Mat compMask = (labels == i);
            blackMask.setTo(255, compMask);
        }
    }

    // ---------- 5. 合并：总保护掩膜 = 彩色 + 黑章 ----------
    cv::Mat mask;
    cv::bitwise_or(colorMask, blackMask, mask);

    // ---------- 6. 膨胀（外扩保护范围） ----------
    if (options.dilateSize > 0) {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(options.dilateSize * 2 + 1, options.dilateSize * 2 + 1));
        cv::dilate(mask, mask, kernel);
    }

    // ---------- 7. 统计 ----------
    const int protectedPixels = cv::countNonZero(mask);

    result.mask = mask;
    result.colorMask = colorMask;
    result.protectedPixels = protectedPixels;
    result.ok = true;
    return result;
}

// ============================================================
// 只取保护掩膜
// ============================================================
cv::Mat StampProtect::makeProtectMask(const cv::Mat &src,
                                       const StampProtectOptions &options)
{
    return detect(src, options).mask;
}

} // namespace protect