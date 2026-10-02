#pragma once

#include <opencv2/core.hpp>

namespace protect {

// ============================================================
// 签名印章保护：参数
// ============================================================
struct StampProtectOptions
{
    // 红色印章（HSV 中的红色跨 0°，所以要两段）
    int redHueLow1   = 0;
    int redHueHigh1  = 12;
    int redHueLow2   = 168;
    int redHueHigh2  = 180;
    int redSatMin    = 80;   // 红章饱和度高
    int redValMin    = 60;

    // 蓝色签名/蓝章
    int blueHueLow   = 90;
    int blueHueHigh  = 135;
    int blueSatMin   = 60;
    int blueValMin   = 50;

    // 通用彩色（其它高饱和度内容，例如绿、紫、青等）
    int colorSatMin  = 70;   // 饱和度阈值
    int colorValMin  = 60;   // 明度阈值（排除接近黑）

    // 黑色印章：黑色低明度，但形状是"密集块"而不是笔画
    bool protectBlackStamp  = true;
    int blackValMax         = 60;    // 视为黑的明度上限
    int blackMinArea        = 800;   // 黑块最小面积
    double blackFillRatio   = 0.55;  // 实心度阈值

    // 保护掩膜膨胀量（保护范围外扩，避免边缘误伤）
    int dilateSize          = 5;

    // 只处理这些颜色通道（默认全开）
    bool enableRed          = true;
    bool enableBlue         = true;
    bool enableOtherColor   = true;
};

// ============================================================
// 签名印章保护：结果
// ============================================================
struct StampProtectResult
{
    cv::Mat mask;            // 8UC1，255=保护区，0=可处理
    cv::Mat colorMask;       // 8UC1，255=彩色区（红/蓝/其它彩色）
    int protectedPixels = 0; // 保护像素数
    bool ok = false;
};

// ============================================================
// 签名印章保护：主类（全部静态方法，无需实例化）
// ============================================================
class StampProtect
{
public:
    // 检测并返回完整结果（含掩膜、彩色掩膜、统计）
    static StampProtectResult detect(const cv::Mat &src,
                                     const StampProtectOptions &options = StampProtectOptions());

    // 只取保护掩膜（供其它处理函数调用）
    static cv::Mat makeProtectMask(const cv::Mat &src,
                                    const StampProtectOptions &options = StampProtectOptions());
};

} // namespace protect