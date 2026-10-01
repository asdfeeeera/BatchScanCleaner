#include "file_scanner.h"

#include <QDir>
#include <QDirIterator>
#include <QFileInfo>

namespace core {

bool FileScanner::isSupportedImage(const QString &path)
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

QStringList FileScanner::scanFolder(const QString &rootPath)
{
    QStringList result;

    QDirIterator it(rootPath,
                    QDir::Files | QDir::NoDotAndDotDot,
                    QDirIterator::Subdirectories);

    while (it.hasNext()) {
        const QString path = it.next();
        if (isSupportedImage(path)) {
            result.append(path);
        }
    }

    return result;
}

} // namespace core