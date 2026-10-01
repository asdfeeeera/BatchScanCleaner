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

    // 确保是 8 位
    cv::Mat toSave;
    if (image.depth() != CV_8U) {
        image.convertTo(toSave, CV_8U);
    } else {
        toSave = image;
    }

    // 编码参数：质量 98 + 4:4:4 色度抽样
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(options.quality);

    if (options.use444Sampling && toSave.channels() == 3) {
        params.push_back(cv::IMWRITE_JPEG_SAMPLING_FACTOR);
        params.push_back(cv::IMWRITE_JPEG_SAMPLING_FACTOR_444);
    }

    std::vector<uchar> buffer;
    if (!cv::imencode(".jpg", toSave, buffer, params)) {
        return false;
    }

    // 用 Qt 写文件，支持中文路径
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

    // 写入 DPI 到 JFIF 密度字段
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

    // 跳过 SOI (FF D8)
    int pos = 2;

    while (pos + 4 < size) {
        const unsigned char marker = static_cast<unsigned char>(data[pos]);
        const unsigned char type   = static_cast<unsigned char>(data[pos + 1]);

        if (marker != 0xFF) {
            break;
        }

        if (type == 0xE0) {
            // APP0 段，检查是否为 JFIF
            if (pos + 9 <= size &&
                data.mid(pos + 4, 5) == QByteArray("JFIF\0", 5)) {

                // JFIF 段布局：
                // pos+0~1: FF E0
                // pos+2~3: 段长度
                // pos+4~8: "JFIF\0"
                // pos+9~10: 版本
                // pos+11: 单位(0=无,1=DPI,2=dpcm)
                // pos+12~13: Xdensity
                // pos+14~15: Ydensity
                const int unitPos = pos + 11;
                const int xPos    = pos + 12;
                const int yPos    = pos + 14;

                if (yPos + 1 < size) {
                    data[unitPos] = 1; // 1 = 每英寸点数

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

        // 其他段：跳过
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