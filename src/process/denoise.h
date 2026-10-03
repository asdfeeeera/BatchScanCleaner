#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace process {

struct DenoiseResult
{
    cv::Mat image;
    int spotCount = 0;
    int cleanedPixels = 0;
    bool ok = false;
    bool skipped = false;

    // ★ 黄色污渍（未自动删除，加到待确认中心）
    std::vector<cv::Rect> yellowBlobs;
};

struct DenoiseOptions
{
    int maxSpotArea = 200;
    int maxSpotWidth = 30;
    int maxSpotHeight = 30;
    double darkRatio = 0.60;
    int protectRadius = 2;
    int strengthLevel = 1;
    bool useInpaint = true;
    cv::Mat protectMask;
};

class Denoise
{
public:
    static DenoiseResult removeSpots(const cv::Mat &src,
                                      const DenoiseOptions &options = DenoiseOptions());

private:
    static double estimatePaperGray(const cv::Mat &gray);

    static void buildProtectMask(const cv::Mat &gray,
                                  cv::Mat &protectMask,
                                  int protectRadius);

    // ★ 检测黄色污渍
    static void detectYellowBlobs(const cv::Mat &src,
                                   std::vector<cv::Rect> &outBlobs);
};

} // namespace process