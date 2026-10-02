#pragma once

#include <opencv2/core.hpp>

namespace protect {

// ============================================================
// Stamp / Signature Protection Options
// ============================================================
struct StampProtectOptions
{
    // --- Red stamp (HSV hue crosses 0, needs two ranges) ---
    int redHueLow1   = 0;
    int redHueHigh1  = 12;
    int redHueLow2   = 168;
    int redHueHigh2  = 180;
    int redSatMin    = 80;
    int redValMin    = 60;

    // --- Blue signature / stamp ---
    int blueHueLow   = 90;
    int blueHueHigh  = 135;
    int blueSatMin   = 60;
    int blueValMin   = 50;

    // --- Other saturated colors ---
    int colorSatMin  = 70;
    int colorValMin  = 60;

    // --- Black stamp (solid black block) ---
    bool protectBlackStamp  = true;
    int blackValMax         = 60;
    int blackMinArea        = 800;
    double blackFillRatio   = 0.55;

    // --- Handwriting signature (thick dark strokes) ---
    bool protectHandwriting  = true;
    int handwritingValMax    = 90;   // gray value <= this is "dark"
    int handwritingOpenSize  = 3;    // opening kernel; strokes wider than this survive
    int handwritingCloseSize = 7;    // closing kernel; connect broken strokes
    int handwritingMinArea   = 300;  // minimum component area
    int handwritingMaxHeight = 400;  // avoid protecting huge text blocks

    // --- Mask dilation (to protect borders) ---
    int dilateSize          = 5;

    // --- Enable / disable per channel ---
    bool enableRed          = true;
    bool enableBlue         = true;
    bool enableOtherColor   = true;
};

// ============================================================
// Result
// ============================================================
struct StampProtectResult
{
    cv::Mat mask;             // 8UC1, 255 = protected
    cv::Mat colorMask;        // 8UC1, 255 = red/blue/other color
    cv::Mat handwritingMask;  // 8UC1, 255 = handwriting signature
    int protectedPixels = 0;
    int colorPixels     = 0;
    int handwritingPixels = 0;
    bool ok = false;
};

// ============================================================
// StampProtect
// ============================================================
class StampProtect
{
public:
    static StampProtectResult detect(const cv::Mat &src,
                                     const StampProtectOptions &options = StampProtectOptions());

    static cv::Mat makeProtectMask(const cv::Mat &src,
                                    const StampProtectOptions &options = StampProtectOptions());
};

} // namespace protect