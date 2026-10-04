#include "backup_dialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>
#include <QSettings>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QDateTime>

namespace backup {

// ============================================================
// 工具函数：人性化大小
// ============================================================
static QString humanSize(qint64 bytes)
{
    const double kb = 1024.0;
    const double mb = kb * 1024.0;
    const double gb = mb * 1024.0;

    const double b = static_cast<double>(bytes);
    if (b < kb) {
        return QString::fromUtf8("%1 B").arg(bytes);
    } else if (b < mb) {
        return QString::fromUtf8("%1 KB").arg(b / kb, 0, 'f', 1);
    } else if (b < gb) {
        return QString::fromUtf8("%1 MB").arg(b / mb, 0, 'f', 2);
    } else {
        return QString::fromUtf8("%1 GB").arg(b / gb, 0, 'f', 2);
    }
}

// ============================================================
// 工具函数：从 settings.ini 读备份根目录
// ============================================================
static QString loadBackupRootFromSettings()
{
    const QString path = QCoreApplication::applicationDirPath()
                         + QStringLiteral("/settings.ini");
    QSettings ini(path, QSettings::IniFormat);
    ini.setIniCodec("UTF-8");

    ini.beginGroup(QStringLiteral("Backup"));
    const QString dir = ini.value(QStringLiteral("backupDir")).toString();
    ini.endGroup();
    return dir;
}

// ============================================================
// 工具函数：统计一个目录里的文件数与总大小（只算文件，不算子目录）
// ============================================================
static void countFilesRecursive(const QString &dirPath,
                                int &fileCount,
                                qint64 &totalBytes)
{
    QDirIterator it(dirPath, QDir::Files | QDir::NoSymLinks,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        it.next();
        const QFileInfo fi = it.fileInfo();
        ++fileCount;
        totalBytes += fi.size();
    }
}

// ============================================================
// 构造
// ============================================================
BackupDialog::BackupDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QString::fromUtf8("备份管理"));
    resize(820, 560);
    setupUi();

    // 从 settings.ini 读默认根目录
    const QString root = loadBackupRootFromSettings();
    m_rootEdit->setText(root);

    loadBackupList();
}

BackupDialog::~BackupDialog() = default;

// ============================================================
// UI
// ============================================================
void BackupDialog::setupUi()
{
    // ---- 顶部：根目录 ----
    QLabel *rootLabel = new QLabel(QString::fromUtf8("备份根目录："), this);

    m_rootEdit = new QLineEdit(this);
    m_rootEdit->setReadOnly(true);
    m_rootEdit->setPlaceholderText(
        QString::fromUtf8("未设置。请到“设置 → 批处理 → 备份目录”里配置，或点右侧选择。"));

    m_browseBtn = new QPushButton(QString::fromUtf8("选择..."), this);
    connect(m_browseBtn, &QPushButton::clicked,
            this, &BackupDialog::onBrowseRoot);

    m_refreshBtn = new QPushButton(QString::fromUtf8("刷新"), this);
    connect(m_refreshBtn, &QPushButton::clicked,
            this, &BackupDialog::onRefresh);

    QHBoxLayout *rootLayout = new QHBoxLayout();
    rootLayout->addWidget(rootLabel);
    rootLayout->addWidget(m_rootEdit, 1);
    rootLayout->addWidget(m_browseBtn);
    rootLayout->addWidget(m_refreshBtn);

    // ---- 中间：表格 ----
    m_table = new QTableWidget(this);
    m_table->setColumnCount(4);
    m_table->setHorizontalHeaderLabels({
        QString::fromUtf8("日期"),
        QString::fromUtf8("文件数"),
        QString::fromUtf8("总大小"),
        QString::fromUtf8("完整路径")
    });
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setSelectionMode(QAbstractItemView::SingleSelection);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->verticalHeader()->setVisible(false);
    m_table->horizontalHeader()->setStretchLastSection(true);
    m_table->setColumnWidth(0, 120);
    m_table->setColumnWidth(1, 80);
    m_table->setColumnWidth(2, 100);

    connect(m_table, &QTableWidget::cellDoubleClicked,
            this, &BackupDialog::onTableDoubleClicked);

    // ---- 底部：汇总 + 按钮 ----
    m_summaryLabel = new QLabel(QString::fromUtf8("就绪"), this);

    m_openBtn = new QPushButton(QString::fromUtf8("打开目录"), this);
    m_deleteBtn = new QPushButton(QString::fromUtf8("删除选中"), this);
    m_clearBtn = new QPushButton(QString::fromUtf8("全部清空"), this);
    m_closeBtn = new QPushButton(QString::fromUtf8("关闭"), this);

    connect(m_openBtn, &QPushButton::clicked,
            this, &BackupDialog::onOpenSelected);
    connect(m_deleteBtn, &QPushButton::clicked,
            this, &BackupDialog::onDeleteSelected);
    connect(m_clearBtn, &QPushButton::clicked,
            this, &BackupDialog::onClearAll);
    connect(m_closeBtn, &QPushButton::clicked,
            this, &QDialog::close);

    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLayout->addWidget(m_summaryLabel, 1);
    btnLayout->addWidget(m_openBtn);
    btnLayout->addWidget(m_deleteBtn);
    btnLayout->addWidget(m_clearBtn);
    btnLayout->addWidget(m_closeBtn);

    // ---- 主布局 ----
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(rootLayout);
    mainLayout->addWidget(m_table, 1);
    mainLayout->addLayout(btnLayout);
}

// ============================================================
// 取当前根目录
// ============================================================
QString BackupDialog::currentRootDir() const
{
    return m_rootEdit->text().trimmed();
}

// ============================================================
// 扫描并填充表格
// ============================================================
void BackupDialog::loadBackupList()
{
    m_table->setRowCount(0);

    const QString root = currentRootDir();
    if (root.isEmpty()) {
        m_summaryLabel->setText(
            QString::fromUtf8("未设置备份根目录。请到“设置”里配置，或点“选择...”"));
        return;
    }

    QDir rootDir(root);
    if (!rootDir.exists()) {
        m_summaryLabel->setText(
            QString::fromUtf8("备份根目录不存在：%1").arg(root));
        return;
    }

    // 找所有 YYYY-MM-DD 子目录
    const QStringList subDirs =
        rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);

    struct Row
    {
        QString date;
        QString fullPath;
        int     fileCount = 0;
        qint64  totalBytes = 0;
    };
    QList<Row> rows;

    QRegExp dateRe(QStringLiteral("^\\d{4}-\\d{2}-\\d{2}$"));
    for (const QString &sub : subDirs) {
        if (!dateRe.exactMatch(sub)) continue;

        Row r;
        r.date = sub;
        r.fullPath = rootDir.filePath(sub);

        countFilesRecursive(r.fullPath, r.fileCount, r.totalBytes);
        rows.append(r);
    }

    // 按日期倒序（最新在最上）
    std::sort(rows.begin(), rows.end(),
              [](const Row &a, const Row &b) {
                  return a.date > b.date;
              });

    m_table->setRowCount(rows.size());
    int totalFiles = 0;
    qint64 totalBytes = 0;
    for (int i = 0; i < rows.size(); ++i) {
        const Row &r = rows[i];
        totalFiles += r.fileCount;
        totalBytes += r.totalBytes;

        auto *c0 = new QTableWidgetItem(r.date);
        auto *c1 = new QTableWidgetItem(QString::number(r.fileCount));
        auto *c2 = new QTableWidgetItem(humanSize(r.totalBytes));
        auto *c3 = new QTableWidgetItem(r.fullPath);
        c3->setToolTip(r.fullPath);

        // 把完整路径存到第 0 列的 UserRole，方便取出
        c0->setData(Qt::UserRole, r.fullPath);

        m_table->setItem(i, 0, c0);
        m_table->setItem(i, 1, c1);
        m_table->setItem(i, 2, c2);
        m_table->setItem(i, 3, c3);
    }

    m_summaryLabel->setText(
        QString::fromUtf8("共 %1 个日期目录，%2 个文件，合计 %3")
            .arg(rows.size())
            .arg(totalFiles)
            .arg(humanSize(totalBytes)));
}

void BackupDialog::updateSummary()
{
    // 简单重扫一遍（也可以不重扫，保持现状）
    loadBackupList();
}

// ============================================================
// 槽：刷新
// ============================================================
void BackupDialog::onRefresh()
{
    loadBackupList();
}

// ============================================================
// 槽：选择根目录
// ============================================================
void BackupDialog::onBrowseRoot()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this,
        QString::fromUtf8("选择备份根目录"),
        currentRootDir(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (dir.isEmpty()) return;

    m_rootEdit->setText(dir);
    loadBackupList();
}

// ============================================================
// 槽：打开选中目录
// ============================================================
void BackupDialog::onOpenSelected()
{
    const int row = m_table->currentRow();
    if (row < 0) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
            QString::fromUtf8("请先选中一行。"));
        return;
    }

    QTableWidgetItem *c0 = m_table->item(row, 0);
    if (!c0) return;
    const QString path = c0->data(Qt::UserRole).toString();
    if (path.isEmpty()) return;

    if (!QFileInfo::exists(path)) {
        QMessageBox::warning(this, QString::fromUtf8("提示"),
            QString::fromUtf8("目录不存在：\n%1").arg(path));
        return;
    }

    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

// ============================================================
// 槽：删除选中
// ============================================================
void BackupDialog::onDeleteSelected()
{
    const int row = m_table->currentRow();
    if (row < 0) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
            QString::fromUtf8("请先选中一行。"));
        return;
    }

    QTableWidgetItem *c0 = m_table->item(row, 0);
    if (!c0) return;
    const QString date = c0->text();
    const QString path = c0->data(Qt::UserRole).toString();

    const auto ret = QMessageBox::question(
        this,
        QString::fromUtf8("确认删除"),
        QString::fromUtf8("确定要删除这一天的备份吗？\n\n"
                          "日期：%1\n路径：%2\n\n"
                          "删除后无法恢复。")
            .arg(date, path));

    if (ret != QMessageBox::Yes) return;

    QDir dir(path);
    if (!dir.removeRecursively()) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
            QString::fromUtf8("删除失败：\n%1").arg(path));
        return;
    }

    loadBackupList();
}

// ============================================================
// 槽：全部清空（保留根目录，删掉里面所有日期子目录）
// ============================================================
void BackupDialog::onClearAll()
{
    const QString root = currentRootDir();
    if (root.isEmpty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
            QString::fromUtf8("未设置备份根目录。"));
        return;
    }

    if (m_table->rowCount() == 0) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
            QString::fromUtf8("没有可清空的备份。"));
        return;
    }

    const auto ret = QMessageBox::warning(
        this,
        QString::fromUtf8("确认清空"),
        QString::fromUtf8("确定要删除根目录下所有日期备份吗？\n\n"
                          "根目录：%1\n"
                          "共 %2 个日期目录会被删除。\n\n"
                          "删除后无法恢复！")
            .arg(root)
            .arg(m_table->rowCount()),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (ret != QMessageBox::Yes) return;

    QDir rootDir(root);
    const QStringList subDirs =
        rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);

    QRegExp dateRe(QStringLiteral("^\\d{4}-\\d{2}-\\d{2}$"));
    int failed = 0;
    for (const QString &sub : subDirs) {
        if (!dateRe.exactMatch(sub)) continue;

        QDir d(rootDir.filePath(sub));
        if (!d.removeRecursively()) {
            ++failed;
        }
    }

    if (failed > 0) {
        QMessageBox::warning(this, QString::fromUtf8("提示"),
            QString::fromUtf8("有 %1 个目录删除失败（可能被占用）。").arg(failed));
    }

    loadBackupList();
}

// ============================================================
// 槽：双击表格 = 打开目录
// ============================================================
void BackupDialog::onTableDoubleClicked(int row, int column)
{
    Q_UNUSED(column);
    if (row < 0) return;

    QTableWidgetItem *c0 = m_table->item(row, 0);
    if (!c0) return;
    const QString path = c0->data(Qt::UserRole).toString();
    if (path.isEmpty()) return;

    if (QFileInfo::exists(path)) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    }
}

} // namespace backup