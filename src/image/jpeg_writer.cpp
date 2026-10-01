#include "jpeg_writer.h"

#include <QFile>
#include <QByteArray>
#include <vector>
#include <opencv2/imgcodecs.hpp>

namespace image {

bool JpegWriter::write(const QString &path,
                       const cv::Mat &image,
                       const JpegSaveOptions &options)
{
    if (image.empty()) {
        return false;
    }

    cv::Mat toSave;
    if (image.depth() != CV_8U) {
        image.convertTo(toSave, CV_8U);
    } else {
        toSave = image;
    }

    // 编码参数：仅设置 JPG 质量
    // OpenCV imencode 默认使用 4:4:4 色度抽样，不会降低彩色细节
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(options.quality);

    std::vector<uchar> buffer;
    if (!cv::imencode(".jpg", toSave, buffer, params)) {
        return false;
    }

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    const qint64 written = file.write(reinterpret_cast<const char *>(buffer.data()),
                                      static_cast<qint64>(buffer.size()));
    file.close();

    if (written != static_cast<qint64>(buffer.size())) {
        return false;
    }

    if (options.dpiX > 0 && options.dpiY > 0) {
        patchJfifDensity(path, options.dpiX, options.dpiY);
    }

    return true;
}

bool JpegWriter::patchJfifDensity(const QString &path, double dpiX, double dpiY)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadWrite)) {
        return false;
    }

    QByteArray data = file.readAll();
    const int size = data.size();

    int pos = 2;

    while (pos + 4 < size) {
        const unsigned char marker = static_cast<unsigned char>(data[pos]);
        const unsigned char type   = static_cast<unsigned char>(data[pos + 1]);

        if (marker != 0xFF) {
            break;
        }

        if (type == 0xE0) {
            if (pos + 9 <= size &&
                data.mid(pos + 4, 5) == QByteArray("JFIF\0", 5)) {

                const int unitPos = pos + 11;
                const int xPos    = pos + 12;
                const int yPos    = pos + 14;

                if (yPos + 1 < size) {
                    data[unitPos] = 1;

                    const unsigned short xd = static_cast<unsigned short>(dpiX + 0.5);
                    const unsigned short yd = static_cast<unsigned short>(dpiY + 0.5);

                    data[xPos]     = static_cast<char>((xd >> 8) & 0xFF);
                    data[xPos + 1] = static_cast<char>(xd & 0xFF);
                    data[yPos]     = static_cast<char>((yd >> 8) & 0xFF);
                    data[yPos + 1] = static_cast<char>(yd & 0xFF);

                    file.seek(0);
                    file.write(data);
                    file.close();
                    return true;
                }
            }
            break;
        }

        if (pos + 4 > size) {
            break;
        }
        const unsigned short segLen =
            (static_cast<unsigned char>(data[pos + 2]) << 8) |
            static_cast<unsigned char>(data[pos + 3]);
        pos += 2 + segLen;
    }

    file.close();
    return false;
}

} // namespace image