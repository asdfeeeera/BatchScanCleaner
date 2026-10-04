#pragma once

#include <QDialog>
#include <QString>

class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;

namespace backup {

// ============================================================
// 备份管理窗口
//   列出 <备份目录>/YYYY-MM-DD/ 下所有日期目录，
//   支持打开、删除单个日期、清空全部。
// ============================================================
class BackupDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BackupDialog(QWidget *parent = nullptr);
    ~BackupDialog() override;

private slots:
    void onRefresh();
    void onBrowseRoot();
    void onOpenSelected();
    void onDeleteSelected();
    void onClearAll();
    void onTableDoubleClicked(int row, int column);

private:
    void setupUi();
    void loadBackupList();
    void updateSummary();
    QString currentRootDir() const;

    QLineEdit    *m_rootEdit  = nullptr;
    QPushButton  *m_browseBtn = nullptr;
    QPushButton  *m_refreshBtn = nullptr;

    QTableWidget *m_table     = nullptr;

    QLabel       *m_summaryLabel = nullptr;

    QPushButton  *m_openBtn    = nullptr;
    QPushButton  *m_deleteBtn  = nullptr;
    QPushButton  *m_clearBtn   = nullptr;
    QPushButton  *m_closeBtn   = nullptr;
};

} // namespace backup