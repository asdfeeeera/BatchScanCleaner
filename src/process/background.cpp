#include "background.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

// ============================================================
// Estimate paper gray (relative to whole image)
// ============================================================
double Background::estimatePaperGray(const cv::Mat &gray,
                                      const BackgroundOptions &options)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int mx = static_cast<int>(W * (1.0 - options.paperSampleRatio) / 2.0);
    const int my = static_cast<int>(H * (1.0 - options.paperSampleRatio) / 2.0);

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

    const size_t idx = static_cast<size_t>(pixels.size() * options.paperPercentile);
    const size_t safeIdx = std::min(idx, pixels.size() - 1);
    std::nth_element(pixels.begin(), pixels.begin() + safeIdx, pixels.end());
    return static_cast<double>(pixels[safeIdx]);
}

// ============================================================
// Main whitening
// ============================================================
BackgroundResult Background::whiten(const cv::Mat &src,
                                     const BackgroundOptions &options)
{
    BackgroundResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;
    const int channels = src.channels();

    // Convert to gray
    cv::Mat gray;
    if (channels == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (channels == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    // Estimate paper gray
    const double paperGray = estimatePaperGray(gray, options);
    result.paperGray = paperGray;

    // Paper already very white -> skip
    if (paperGray >= 254.5) {
        result.image = src.clone();
        result.skipped = true;
        result.ok = true;
        return result;
    }

    // Content mask: darker than paperGray * contentRatio
    const double contentThreshold = paperGray * options.contentRatio;

    cv::Mat contentMask;
    cv::threshold(gray, contentMask, contentThreshold, 255, cv::THRESH_BINARY_INV);

    // Protect colored content (red stamps, blue signatures)
    if (options.protectColor && channels >= 3) {
        cv::Mat bgr;
        if (channels == 4) {
            cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        } else {
            bgr = src;
        }

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        const cv::Mat &S = hsvCh[1];
        const cv::Mat &V = hsvCh[2];

        cv::Mat colorMask;
        cv::inRange(S, options.colorSatMin, 255, colorMask);

        cv::Mat valMask;
        cv::inRange(V, 40, 255, valMask);
        cv::bitwise_and(colorMask, valMask, colorMask);

        cv::bitwise_or(contentMask, colorMask, contentMask);
    }

    // External protect mask (stamp/signature)
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
        cv::bitwise_or(contentMask, ext, contentMask);
    }

    // Background mask = NOT content
    cv::Mat bgMask;
    cv::bitwise_not(contentMask, bgMask);

    // Whitening
    cv::Mat dst = src.clone();
    cv::Scalar white;
    const int tg = options.targetPaperGray;

    if (channels == 4) {
        white = cv::Scalar(tg, tg, tg, 255);
    } else if (channels == 3) {
        white = cv::Scalar(tg, tg, tg);
    } else {
        white = cv::Scalar(tg);
    }

    dst.setTo(white, bgMask);

    result.image = dst;
    result.whitenedPixels = cv::countNonZero(bgMask);
    result.ok = true;
    return result;
}

} // namespace process