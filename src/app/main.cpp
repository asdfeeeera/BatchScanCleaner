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

    QApplication::setApplicationName(QString::fromUtf8("批量扫描图片净化增强软件"));
    QApplication::setApplicationVersion(QString::fromUtf8("1.0.0.0"));
    QApplication::setOrganizationName(QString::fromUtf8("BatchScanCleaner"));

    QIcon icon(QString::fromUtf8(":/icons/app.ico"));
    if (!icon.isNull()) {
        QApplication::setWindowIcon(icon);
    }

    MainWindow window;
    window.show();

    return app.exec();
}