#pragma once

#include <opencv2/core.hpp>

namespace process {

struct BlackEdgeResult
{
    cv::Mat image;
    int topPixels = 0;
    int bottomPixels = 0;
    int leftPixels = 0;
    int rightPixels = 0;
    int filledPixels = 0;
    double paperGray = 0.0;
    double darkThreshold = 0.0;
    bool ok = false;
    bool skipped = false;
};

struct BlackEdgeOptions
{
    double paperSampleRatio = 0.6;
    double paperPercentile = 0.9;
    double darkRatio = 0.75;
    double maxScanRatio = 0.30;
    double darkPixelRatio = 0.50;
    int gapTolerance = 3;
    int smoothKernelSize = 5;
    int expandPixels = 5;
    bool fillWhite = true;
};

class BlackEdge
{
public:
    static BlackEdgeResult removeBlackEdge(const cv::Mat &src,
                                            const BlackEdgeOptions &options = BlackEdgeOptions());

private:
    static double estimatePaperGray(const cv::Mat &gray, const BlackEdgeOptions &options);
    static int scanTop(const cv::Mat &gray, double darkThreshold, const BlackEdgeOptions &options);
    static int scanBottom(const cv::Mat &gray, double darkThreshold, const BlackEdgeOptions &options);
    static int scanLeft(const cv::Mat &gray, double darkThreshold, const BlackEdgeOptions &options);
    static int scanRight(const cv::Mat &gray, double darkThreshold, const BlackEdgeOptions &options);
    static void smoothMask(cv::Mat &mask, int kernelSize);
};

} // namespace process