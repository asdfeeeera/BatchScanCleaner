#include <QApplication>
#include <QIcon>
#include <QMetaType>
#include <QSettings>
#include <QCoreApplication>
#include <QString>
#include "main_window.h"
#include "../db/batch_processor.h"
#include "../ui/wizard_dialog.h"

int main(int argc, char *argv[])
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps, true);
#endif

    QApplication app(argc, argv);

    // ★ 关键：注册跨线程信号用的自定义类型
    //   短名和全名都要注册，因为 moc 生成的签名用短名。
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

    // ★ 3.4 首次启动向导
    //   检查 settings.ini 里的 [FirstRun] completed 标记；
    //   没有标记 → 弹出向导；点“完成”或“跳过”都会写标记，以后不再弹。
    {
        const QString iniPath = QCoreApplication::applicationDirPath()
                                + QStringLiteral("/settings.ini");
        QSettings ini(iniPath, QSettings::IniFormat);
        ini.setIniCodec("UTF-8");
        ini.beginGroup(QStringLiteral("FirstRun"));
        const bool completed =
            ini.value(QStringLiteral("completed"), 0).toInt() != 0;
        ini.endGroup();

        if (!completed) {
            ui::WizardDialog wizard(&window);
            wizard.exec();
        }
    }

    return app.exec();
}