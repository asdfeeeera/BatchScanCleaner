#pragma once

#include <opencv2/core.hpp>

namespace process {

struct DeskewResult
{
    cv::Mat image;      // 扶正后的图像
    double angle = 0.0; // 检测到的倾斜角（度，正数=逆时针）
    bool ok = false;    // 是否成功扶正
};

class Deskew
{
public:
    // 自动检测倾斜角并扶正
    // 输入：BGR 或灰度图
    // 输出：白底扩展、不裁正文、不改变像素密度
    static DeskewResult autoDeskew(const cv::Mat &src);

    // 只检测倾斜角，不旋转
    static double detectAngle(const cv::Mat &src);

private:
    // 用旋转后的画布扩展算法，保证内容完整
    static cv::Mat rotateKeepAll(const cv::Mat &src, double angle);
};

} // namespace process