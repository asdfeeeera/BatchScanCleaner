#include "stamp_protect.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace protect {

namespace {

// Ensure src is 3-channel BGR
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

// ============================================================
// Detect handwriting signatures
// Idea: handwriting strokes are thicker (3-8 px) than
// printed text strokes (1-2 px). Use morphological opening
// to remove thin strokes, keep thick ones.
// ============================================================
cv::Mat detectHandwriting(const cv::Mat &bgr,
                           const StampProtectOptions &opt)
{
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // Dark pixels -> white
    cv::Mat darkBin;
    cv::threshold(gray, darkBin, opt.handwritingValMax, 255,
                  cv::THRESH_BINARY_INV);

    // Opening: remove thin printed strokes
    cv::Mat kOpen = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(opt.handwritingOpenSize, opt.handwritingOpenSize));
    cv::Mat opened;
    cv::morphologyEx(darkBin, opened, cv::MORPH_OPEN, kOpen);

    // Closing: connect broken handwriting strokes
    cv::Mat kClose = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(opt.handwritingCloseSize, opt.handwritingCloseSize));
    cv::Mat closed;
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, kClose);

    // Connected components: filter by size
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(
        closed, labels, stats, centroids, 8, CV_32S);

    cv::Mat result = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8UC1);

    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (area < opt.handwritingMinArea) continue;
        if (h > opt.handwritingMaxHeight) continue;
        if (w > bgr.cols / 2) continue;  // too wide, probably a table line

        cv::Mat compMask = (labels == i);
        result.setTo(255, compMask);
    }

    return result;
}

} // namespace

// ============================================================
// Main detect
// ============================================================
StampProtectResult StampProtect::detect(const cv::Mat &src,
                                         const StampProtectOptions &options)
{
    StampProtectResult result;
    if (src.empty()) return result;

    const cv::Mat bgr = toBgr(src);
    const int W = bgr.cols;
    const int H = bgr.rows;

    // Convert to HSV
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvCh;
    cv::split(hsv, hsvCh);
    const cv::Mat &Hc = hsvCh[0];
    const cv::Mat &Sc = hsvCh[1];
    const cv::Mat &Vc = hsvCh[2];

    cv::Mat colorMask = cv::Mat::zeros(H, W, CV_8UC1);

    // ---------- 1. Red ----------
    if (options.enableRed) {
        cv::Mat red1, red2, redMask;
        cv::inRange(Hc, options.redHueLow1, options.redHueHigh1, red1);
        cv::inRange(Hc, options.redHueLow2, options.redHueHigh2, red2);
        cv::bitwise_or(red1, red2, redMask);

        cv::Mat redSat, redVal;
        cv::inRange(Sc, options.redSatMin, 255, redSat);
        cv::inRange(Vc, options.redValMin, 255, redVal);
        cv::bitwise_and(redMask, redSat, redMask);
        cv::bitwise_and(redMask, redVal, redMask);
        cv::bitwise_or(colorMask, redMask, colorMask);
    }

    // ---------- 2. Blue ----------
    if (options.enableBlue) {
        cv::Mat blueMask, blueSat, blueVal;
        cv::inRange(Hc, options.blueHueLow, options.blueHueHigh, blueMask);
        cv::inRange(Sc, options.blueSatMin, 255, blueSat);
        cv::inRange(Vc, options.blueValMin, 255, blueVal);
        cv::bitwise_and(blueMask, blueSat, blueMask);
        cv::bitwise_and(blueMask, blueVal, blueMask);
        cv::bitwise_or(colorMask, blueMask, colorMask);
    }

    // ---------- 3. Other saturated colors ----------
    if (options.enableOtherColor) {
        cv::Mat sat, val, otherMask;
        cv::inRange(Sc, options.colorSatMin, 255, sat);
        cv::inRange(Vc, options.colorValMin, 255, val);
        cv::bitwise_and(sat, val, otherMask);
        cv::bitwise_or(colorMask, otherMask, colorMask);
    }

    // ---------- 4. Black stamp (solid black blocks) ----------
    cv::Mat blackMask = cv::Mat::zeros(H, W, CV_8UC1);
    if (options.protectBlackStamp) {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        cv::Mat blackBin;
        cv::threshold(gray, blackBin, options.blackValMax, 255,
                      cv::THRESH_BINARY_INV);

        cv::Mat openKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(9, 9));
        cv::Mat opened;
        cv::morphologyEx(blackBin, opened, cv::MORPH_OPEN, openKernel);

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

            cv::Mat compMask = (labels == i);
            blackMask.setTo(255, compMask);
        }
    }

    // ---------- 5. Handwriting signature ----------
    cv::Mat handwritingMask;
    if (options.protectHandwriting) {
        handwritingMask = detectHandwriting(bgr, options);
    } else {
        handwritingMask = cv::Mat::zeros(H, W, CV_8UC1);
    }

    // ---------- 6. Merge ----------
    cv::Mat mask;
    cv::bitwise_or(colorMask, blackMask, mask);
    cv::bitwise_or(mask, handwritingMask, mask);

    // ---------- 7. Dilation ----------
    if (options.dilateSize > 0) {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(options.dilateSize * 2 + 1, options.dilateSize * 2 + 1));
        cv::dilate(mask, mask, kernel);
    }

    // ---------- 8. Stats ----------
    result.mask = mask;
    result.colorMask = colorMask;
    result.handwritingMask = handwritingMask;
    result.protectedPixels = cv::countNonZero(mask);
    result.colorPixels = cv::countNonZero(colorMask);
    result.handwritingPixels = cv::countNonZero(handwritingMask);
    result.ok = true;
    return result;
}

cv::Mat StampProtect::makeProtectMask(const cv::Mat &src,
                                       const StampProtectOptions &options)
{
    return detect(src, options).mask;
}

} // namespace protect