#pragma once

#include <QString>
#include <QStringList>

namespace core {

class FileScanner
{
public:
    // 递归扫描文件夹，返回所有支持的图片文件路径
    static QStringList scanFolder(const QString &rootPath);

    // 判断是否为支持的图片格式
    static bool isSupportedImage(const QString &path);
};

} // namespace core