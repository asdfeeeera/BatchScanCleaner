#include <QApplication>
#include <QIcon>
#include "main_window.h"

int main(int argc, char *argv[])
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps, true);
#endif

    QApplication app(argc, argv);

    QApplication::setApplicationName(QStringLiteral("批量扫描图片净化增强软件"));
    QApplication::setApplicationVersion(QStringLiteral("1.0.0.0"));
    QApplication::setOrganizationName(QStringLiteral("BatchScanCleaner"));

    QIcon icon(QStringLiteral(":/icons/app.ico"));
    if (!icon.isNull()) {
        QApplication::setWindowIcon(icon);
    }

    MainWindow window;
    window.show();

    return app.exec();
}