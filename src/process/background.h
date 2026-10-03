#pragma once

#include <opencv2/core.hpp>

namespace process {

// ============================================================
// Background whitening options
// ============================================================
struct BackgroundOptions
{
    // Paper gray estimation: use percentile in center region
    double paperSampleRatio = 0.6;    // use 60% center
    double paperPercentile  = 0.9;    // 90th percentile = paper gray

    // Strong whitening: paper pixels above this -> white
    // 0.0 ~ 1.0 (relative to estimated paper gray)
    // 0.85 = pixels darker than 85% of paper gray are "content"
    double contentRatio = 0.85;

    // Target paper gray (255 = pure white)
    int targetPaperGray = 255;

    // Protect colored content (red stamps, blue signatures)
    bool protectColor = true;
    int colorSatMin = 40;   // HSV saturation threshold

    // Protect external mask (from stamp/signature protection)
    cv::Mat protectMask;
};

// ============================================================
// Background whitening result
// ============================================================
struct BackgroundResult
{
    cv::Mat image;
    int whitenedPixels = 0;
    double paperGray = 0.0;
    bool ok = false;
    bool skipped = false;   // paper already white
};

// ============================================================
// Background whitening class
// ============================================================
class Background
{
public:
    static BackgroundResult whiten(const cv::Mat &src,
                                    const BackgroundOptions &options = BackgroundOptions());

private:
    static double estimatePaperGray(const cv::Mat &gray,
                                     const BackgroundOptions &options);
};

} // namespace process