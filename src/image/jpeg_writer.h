#pragma once

#include <QString>
#include <opencv2/core.hpp>

namespace image {

struct JpegSaveOptions
{
    int quality = 98;         // JPG 质量 98–100
    double dpiX = 300.0;      // 输出 DPI
    double dpiY = 300.0;
    bool use444Sampling = true; // 4:4:4 色度抽样，保留红章蓝签
};

class JpegWriter
{
public:
    // 保存 JPG，保持 DPI、像素、4:4:4 采样、高质量
    static bool write(const QString &path,
                      const cv::Mat &image,
                      const JpegSaveOptions &options);

private:
    // 修改 JPG 文件头的 JFIF 密度字段（DPI）
    static bool patchJfifDensity(const QString &path, double dpiX, double dpiY);
};

} // namespace image