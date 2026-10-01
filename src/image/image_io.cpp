#include "image_io.h"

#include <QFile>
#include <QFileInfo>
#include <vector>
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
    // 用 Qt 读取文件字节流，避免 OpenCV 中文路径问题
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    const QByteArray data = file.readAll();
    file.close();

    if (data.isEmpty()) {
        return false;
    }

    std::vector<uchar> buffer(data.begin(), data.end());
    cv::Mat mat = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (mat.empty()) {
        return false;
    }

    outImage = mat;
    outMeta.sourcePath = path;
    outMeta.width = mat.cols;
    outMeta.height = mat.rows;
    outMeta.channels = mat.channels();
    return true;
}

} // namespace image