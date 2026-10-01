#pragma once

#include <QString>
#include <opencv2/core.hpp>

namespace image {

struct ImageMeta
{
    QString sourcePath;
    int width = 0;
    int height = 0;
    int channels = 0;
};

class ImageIO
{
public:
    static bool isSupportedExtension(const QString &path);
    static bool read(const QString &path, cv::Mat &outImage, ImageMeta &outMeta);
};

} // namespace image