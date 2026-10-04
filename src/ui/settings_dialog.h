#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>

class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QListWidget;
class QGroupBox;

namespace ui {

// ============================================================
// 应用设置（存于 <程序目录>/settings.ini）
// ============================================================
struct AppSettings
{
    // ① 通用
    QString defaultOutputDir;
    QString defaultOutputSuffix;
    int     defaultJpegQuality = 98;
    double  defaultDpi         = 300.0;   // ★ 新增：默认 DPI

    // ② Tesseract（错误页码处理）
    QString tesseractPath;
    bool    errPageTopLeft = true;
    bool    errPageTopRight = true;
    bool    errPageBottomRight = true;
    int     errPageMinDigitHeight = 20;
    int     errPageMaxDigitHeight = 100;

    // ③ 批处理
    QStringList defaultExtraDirs;
    int     defaultShardIndex = 0;
    int     defaultTotalShards = 1;
    QString reportDir;

    // 读写（ini 文件固定为 <程序目录>/settings.ini）
    static QString filePath();
    static AppSettings load();
    void save() const;
};

// ============================================================
// 设置窗口
// ============================================================
class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = nullptr);
    ~SettingsDialog() override;

private slots:
    void onBrowseOutputDir();
    void onBrowseTesseract();
    void onBrowseReportDir();
    void onAddExtraDir();
    void onRemoveExtraDir();
    void onResetDefaults();
    void onOk();
    void onApply();

private:
    void setupUi();
    void loadToUi();
    void uiToSettings(AppSettings &s) const;
    void applySettings();

    // 通用
    QLineEdit   *m_outputDirEdit = nullptr;
    QPushButton *m_outputDirBtn  = nullptr;
    QLineEdit   *m_suffixEdit    = nullptr;
    QSpinBox    *m_jpegQualitySpin = nullptr;
    QDoubleSpinBox *m_dpiSpin    = nullptr;   // ★ 新增

    // Tesseract
    QLineEdit   *m_tesseractEdit = nullptr;
    QPushButton *m_tesseractBtn  = nullptr;
    QCheckBox   *m_topLeftCheck  = nullptr;
    QCheckBox   *m_topRightCheck = nullptr;
    QCheckBox   *m_bottomRightCheck = nullptr;
    QSpinBox    *m_minDigitHeightSpin = nullptr;
    QSpinBox    *m_maxDigitHeightSpin = nullptr;

    // 批处理
    QListWidget *m_extraDirsList     = nullptr;
    QPushButton *m_addExtraDirBtn    = nullptr;
    QPushButton *m_removeExtraDirBtn = nullptr;
    QSpinBox    *m_shardIndexSpin    = nullptr;
    QSpinBox    *m_totalShardsSpin   = nullptr;
    QLineEdit   *m_reportDirEdit     = nullptr;
    QPushButton *m_reportDirBtn      = nullptr;

    // 底部按钮
    QPushButton *m_resetBtn  = nullptr;
    QPushButton *m_okBtn     = nullptr;
    QPushButton *m_cancelBtn = nullptr;
    QPushButton *m_applyBtn  = nullptr;
};

} // namespace ui