#include "image_io.h"

#include <QFileInfo>
#include <opencv2/imgcodecs.hpp>

namespace image {

bool ImageIO::isSupportedExtension(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    static const QStringList supported = {
        QStringLiteral("jpg"),
        QStringLiteral("jpeg"),
        QStringLiteral("png"),
        QStringLiteral("bmp"),
        QStringLiteral("tif"),
        QStringLiteral("tiff")
    };
    return supported.contains(suffix);
}

bool ImageIO::read(const QString &path, cv::Mat &outImage, ImageMeta &outMeta)
{
    cv::Mat buffer = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (buffer.empty()) {
        return false;
    }

    outImage = buffer;
    outMeta.sourcePath = path;
    outMeta.width = buffer.cols;
    outMeta.height = buffer.rows;
    outMeta.channels = buffer.channels();
    return true;
}

} // namespace image