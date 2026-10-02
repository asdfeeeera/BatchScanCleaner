#pragma once

#include <opencv2/core.hpp>

namespace process {

struct DenoiseResult
{
    cv::Mat image;
    int spotCount = 0;          // 检测到的污点数
    int cleanedPixels = 0;      // 修补的像素数
    bool ok = false;
    bool skipped = false;
};

struct DenoiseOptions
{
    // 污点最大面积（像素），超过此值不当污点处理
    int maxSpotArea = 200;

    // 污点最大宽度（像素），超过此值不当污点处理
    int maxSpotWidth = 30;

    // 污点最大高度（像素）
    int maxSpotHeight = 30;

    // 污点灰度阈值：暗于此值才算污点
    // 相对值，等于 纸张灰度 × darkRatio
    double darkRatio = 0.60;

    // 保护半径：距离文字/线条 N 像素内的暗块不当污点
    int protectRadius = 2;

    // 处理强度：保守 / 标准 / 强力
    // 对应 maxSpotArea 的 0.5 / 1.0 / 2.0 倍
    int strengthLevel = 1;  // 0=保守, 1=标准, 2=强力

    // 用哪种方式填充：中值 / 背景
    bool useInpaint = true;
};

class Denoise
{
public:
    static DenoiseResult removeSpots(const cv::Mat &src,
                                      const DenoiseOptions &options = DenoiseOptions());

private:
    // 估算纸张灰度
    static double estimatePaperGray(const cv::Mat &gray);

    // 构建保护掩膜（文字、线条、签名、印章）
    static void buildProtectMask(const cv::Mat &gray,
                                  cv::Mat &protectMask,
                                  int protectRadius);
};

} // namespace process