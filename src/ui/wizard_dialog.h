#pragma once

#include <QDialog>

class QLabel;
class QLineEdit;
class QPushButton;
class QCheckBox;
class QSpinBox;

namespace ui {

// ============================================================
// 首次启动向导
//   单页 Dialog：欢迎 + 常用目录 + Tesseract + 分片
//   点“完成”保存设置并写首次启动标记；
//   点“跳过”只写标记，不改设置。
// ============================================================
class WizardDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WizardDialog(QWidget *parent = nullptr);
    ~WizardDialog() override;

private slots:
    void onBrowseOutputDir();
    void onBrowseBackupDir();
    void onBrowseTesseract();
    void onFinish();
    void onSkip();

private:
    void setupUi();
    void loadCurrentSettings();
    void saveCurrentSettings();
    void markFirstRunDone();

    // 通用
    QLineEdit   *m_outputDirEdit = nullptr;
    QPushButton *m_outputDirBtn  = nullptr;

    // 备份
    QCheckBox   *m_backupCheck   = nullptr;
    QLineEdit   *m_backupDirEdit = nullptr;
    QPushButton *m_backupDirBtn  = nullptr;

    // Tesseract
    QLineEdit   *m_tesseractEdit = nullptr;
    QPushButton *m_tesseractBtn  = nullptr;

    // 分片
    QSpinBox    *m_shardIndexSpin  = nullptr;
    QSpinBox    *m_totalShardsSpin = nullptr;

    // 底部按钮
    QPushButton *m_skipBtn   = nullptr;
    QPushButton *m_finishBtn = nullptr;
};

} // namespace ui