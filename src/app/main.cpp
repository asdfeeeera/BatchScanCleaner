#include <QApplication>
#include <QIcon>
#include <QMetaType>
#include "main_window.h"
#include "../db/batch_processor.h"

int main(int argc, char *argv[])
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps, true);
#endif

    QApplication app(argc, argv);

    // ★ 关键：注册跨线程信号用的自定义类型
    // 不注册的话，BatchProcessor 在工作线程 emit 的 progressChanged / finished
    // 会被 Qt 静默丢弃，导致进度条不动、“打开报告”按钮不亮。
    qRegisterMetaType<batch::BatchProgress>("batch::BatchProgress");
    qRegisterMetaType<batch::BatchResult>("batch::BatchResult");

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