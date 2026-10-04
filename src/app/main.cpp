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

    // ★ 关键：注册跨线程信号用的类型
    //   注意：必须同时注册“短名”和“全名”，因为 Qt 的 moc 生成的
    //   连接签名里用的是短名（BatchProgress），而 Q_DECLARE_METATYPE
    //   注册的是全名（batch::BatchProgress）。两个名字都注册才保险。
    qRegisterMetaType<batch::BatchProgress>("BatchProgress");
    qRegisterMetaType<batch::BatchProgress>("batch::BatchProgress");
    qRegisterMetaType<batch::BatchResult>("BatchResult");
    qRegisterMetaType<batch::BatchResult>("batch::BatchResult");
    qRegisterMetaType<cv::Mat>("cv::Mat");

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