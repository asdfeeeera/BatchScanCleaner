#include "pending_dialog.h"
#include "pending_center.h"

#include <QListWidget>
#include <QLabel>
#include <QTextEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <QFileInfo>
#include <QScrollArea>

#include <opencv2/imgproc.hpp>

namespace analyze {

// ============================================================
// Helper: cv::Mat -> QPixmap
// ============================================================
namespace {

QPixmap matToPixmap(const cv::Mat &mat, int maxW, int maxH)
{
    if (mat.empty()) {
        return QPixmap();
    }

    cv::Mat rgb;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    }

    // Scale down if needed
    cv::Mat scaled;
    if (rgb.cols > maxW || rgb.rows > maxH) {
        const double sx = static_cast<double>(maxW) / rgb.cols;
        const double sy = static_cast<double>(maxH) / rgb.rows;
        const double s = std::min(sx, sy);
        cv::resize(rgb, scaled, cv::Size(), s, s, cv::INTER_AREA);
    } else {
        scaled = rgb;
    }

    QImage qimg(scaled.data, scaled.cols, scaled.rows,
                static_cast<int>(scaled.step), QImage::Format_RGB888);
    return QPixmap::fromImage(qimg.copy());
}

} // namespace

// ============================================================
// Constructor / Destructor
// ============================================================
PendingDialog::PendingDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QString::fromUtf8("待确认中心"));
    resize(1000, 640);

    setupUi();
    refreshFromCenter();
}

PendingDialog::~PendingDialog() = default;

// ============================================================
// UI
// ============================================================
void PendingDialog::setupUi()
{
    // ---- Left: list ----
    m_listWidget = new QListWidget(this);
    m_listWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_listWidget->setMinimumWidth(280);
    connect(m_listWidget, &QListWidget::itemSelectionChanged,
            this, &PendingDialog::onItemSelectionChanged);

    // ---- Right: preview + detail ----
    m_thumbnailLabel = new QLabel(this);
    m_thumbnailLabel->setAlignment(Qt::AlignCenter);
    m_thumbnailLabel->setMinimumSize(360, 240);
    m_thumbnailLabel->setStyleSheet(
        QStringLiteral("QLabel { background-color: #f0f0f0; border: 1px solid #c0c0c0; }"));

    QScrollArea *thumbScroll = new QScrollArea(this);
    thumbScroll->setWidget(m_thumbnailLabel);
    thumbScroll->setWidgetResizable(true);
    thumbScroll->setMinimumHeight(280);

    m_detailEdit = new QTextEdit(this);
    m_detailEdit->setReadOnly(true);
    m_detailEdit->setMinimumHeight(140);

    QWidget *rightWidget = new QWidget(this);
    QVBoxLayout *rightLayout = new QVBoxLayout(rightWidget);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->addWidget(thumbScroll);
    rightLayout->addWidget(new QLabel(QString::fromUtf8("详细信息："), this));
    rightLayout->addWidget(m_detailEdit);

    // ---- Splitter ----
    QSplitter *splitter = new QSplitter(Qt::Horizontal, this);
    splitter->addWidget(m_listWidget);
    splitter->addWidget(rightWidget);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 2);

    // ---- Top: stats ----
    m_statsLabel = new QLabel(this);

    // ---- Bottom: buttons ----
    m_acceptBtn     = new QPushButton(QString::fromUtf8("接受当前"), this);
    m_rejectBtn     = new QPushButton(QString::fromUtf8("拒绝当前"), this);
    m_ignoreBtn     = new QPushButton(QString::fromUtf8("忽略当前"), this);
    m_acceptAllBtn  = new QPushButton(QString::fromUtf8("全部接受"), this);
    m_rejectAllBtn  = new QPushButton(QString::fromUtf8("全部拒绝"), this);
    m_ignoreAllBtn  = new QPushButton(QString::fromUtf8("全部忽略"), this);
    m_clearDecidedBtn = new QPushButton(QString::fromUtf8("清除已决定"), this);
    m_closeBtn      = new QPushButton(QString::fromUtf8("关闭"), this);

    connect(m_acceptBtn,     &QPushButton::clicked, this, &PendingDialog::onAcceptCurrent);
    connect(m_rejectBtn,     &QPushButton::clicked, this, &PendingDialog::onRejectCurrent);
    connect(m_ignoreBtn,     &QPushButton::clicked, this, &PendingDialog::onIgnoreCurrent);
    connect(m_acceptAllBtn,  &QPushButton::clicked, this, &PendingDialog::onAcceptAll);
    connect(m_rejectAllBtn,  &QPushButton::clicked, this, &PendingDialog::onRejectAll);
    connect(m_ignoreAllBtn,  &QPushButton::clicked, this, &PendingDialog::onIgnoreAll);
    connect(m_clearDecidedBtn, &QPushButton::clicked, this, &PendingDialog::onClearDecided);
    connect(m_closeBtn,      &QPushButton::clicked, this, &QDialog::accept);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(m_acceptBtn);
    buttonLayout->addWidget(m_rejectBtn);
    buttonLayout->addWidget(m_ignoreBtn);
    buttonLayout->addSpacing(20);
    buttonLayout->addWidget(m_acceptAllBtn);
    buttonLayout->addWidget(m_rejectAllBtn);
    buttonLayout->addWidget(m_ignoreAllBtn);
    buttonLayout->addStretch();
    buttonLayout->addWidget(m_clearDecidedBtn);
    buttonLayout->addWidget(m_closeBtn);

    // ---- Main layout ----
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(m_statsLabel);
    mainLayout->addWidget(splitter, 1);
    mainLayout->addLayout(buttonLayout);
}

// ============================================================
// Reload data
// ============================================================
void PendingDialog::refreshFromCenter()
{
    rebuildList();
    updateDetailPanel();
    updateButtonsState();
}

void PendingDialog::rebuildList()
{
    m_listWidget->clear();

    const QList<PendingItem> items = PendingCenter::instance().allItems();
    for (const PendingItem &it : items) {
        QListWidgetItem *lwi = new QListWidgetItem(formatListItemText(it));
        lwi->setData(Qt::UserRole, it.id);
        m_listWidget->addItem(lwi);
    }

    const int total = PendingCenter::instance().count();
    const int pending = PendingCenter::instance().pendingCount();
    m_statsLabel->setText(
        QString::fromUtf8("共 %1 项，其中待确认 %2 项")
            .arg(total).arg(pending));
}

QString PendingDialog::formatListItemText(const PendingItem &it) const
{
    QString line = QString::fromUtf8("[%1] %2")
        .arg(it.id)
        .arg(PendingItem::typeToString(it.type));

    if (!it.fileName.isEmpty()) {
        line += QString::fromUtf8(" — %1").arg(it.fileName);
    }

    line += QString::fromUtf8("  (%1)")
        .arg(PendingItem::decisionToString(it.decision));

    return line;
}

// ============================================================
// Detail panel
// ============================================================
void PendingDialog::updateDetailPanel()
{
    const QList<QListWidgetItem *> sel = m_listWidget->selectedItems();
    if (sel.isEmpty()) {
        m_thumbnailLabel->setText(QString::fromUtf8("（未选中任何项）"));
        m_thumbnailLabel->setPixmap(QPixmap());
        m_detailEdit->clear();
        return;
    }

    const int id = sel.first()->data(Qt::UserRole).toInt();
    const PendingItem it = PendingCenter::instance().itemById(id);

    // Thumbnail
    if (!it.thumbnail.empty()) {
        const QPixmap pix = matToPixmap(it.thumbnail, 500, 400);
        m_thumbnailLabel->setPixmap(pix);
        m_thumbnailLabel->setText(QString());
    } else {
        m_thumbnailLabel->setPixmap(QPixmap());
        m_thumbnailLabel->setText(QString::fromUtf8("（无缩略图）"));
    }

    // Detail
    QString text;
    text += QString::fromUtf8("类型：%1\n").arg(PendingItem::typeToString(it.type));
    text += QString::fromUtf8("建议动作：%1\n").arg(PendingItem::actionToString(it.suggestedAction));
    text += QString::fromUtf8("当前决定：%1\n").arg(PendingItem::decisionToString(it.decision));
    text += QString::fromUtf8("来源文件：%1\n").arg(it.sourceImagePath.isEmpty()
                                                    ? QString::fromUtf8("（无）")
                                                    : it.sourceImagePath);
    text += QString::fromUtf8("位置：(x=%1, y=%2, w=%3, h=%4)\n")
                .arg(it.boundingBox.x).arg(it.boundingBox.y)
                .arg(it.boundingBox.width).arg(it.boundingBox.height);
    text += QString::fromUtf8("置信度：%1\n").arg(it.confidence, 0, 'f', 2);
    text += QString::fromUtf8("原因：%1\n").arg(it.reason);
    text += QString::fromUtf8("详情：%1\n").arg(it.detail);
    if (it.createdAt.isValid()) {
        text += QString::fromUtf8("时间：%1\n")
                    .arg(it.createdAt.toString(QStringLiteral("yyyy-MM-dd HH:mm:ss")));
    }

    m_detailEdit->setPlainText(text);
}

// ============================================================
// Buttons state
// ============================================================
void PendingDialog::updateButtonsState()
{
    const bool hasSelection = !m_listWidget->selectedItems().isEmpty();
    m_acceptBtn->setEnabled(hasSelection);
    m_rejectBtn->setEnabled(hasSelection);
    m_ignoreBtn->setEnabled(hasSelection);

    const bool hasAny = PendingCenter::instance().count() > 0;
    const bool hasPending = PendingCenter::instance().pendingCount() > 0;
    m_acceptAllBtn->setEnabled(hasPending);
    m_rejectAllBtn->setEnabled(hasPending);
    m_ignoreAllBtn->setEnabled(hasPending);
    m_clearDecidedBtn->setEnabled(hasAny);
}

// ============================================================
// Slots
// ============================================================
void PendingDialog::onItemSelectionChanged()
{
    updateDetailPanel();
    updateButtonsState();
}

void PendingDialog::onAcceptCurrent()
{
    QList<int> ids;
    for (QListWidgetItem *lwi : m_listWidget->selectedItems()) {
        ids.append(lwi->data(Qt::UserRole).toInt());
    }
    if (ids.isEmpty()) return;

    PendingCenter::instance().setDecisionByIds(ids, PendingDecision::Accepted);
    refreshFromCenter();
}

void PendingDialog::onRejectCurrent()
{
    QList<int> ids;
    for (QListWidgetItem *lwi : m_listWidget->selectedItems()) {
        ids.append(lwi->data(Qt::UserRole).toInt());
    }
    if (ids.isEmpty()) return;

    PendingCenter::instance().setDecisionByIds(ids, PendingDecision::Rejected);
    refreshFromCenter();
}

void PendingDialog::onIgnoreCurrent()
{
    QList<int> ids;
    for (QListWidgetItem *lwi : m_listWidget->selectedItems()) {
        ids.append(lwi->data(Qt::UserRole).toInt());
    }
    if (ids.isEmpty()) return;

    PendingCenter::instance().setDecisionByIds(ids, PendingDecision::Ignored);
    refreshFromCenter();
}

void PendingDialog::onAcceptAll()
{
    const int n = PendingCenter::instance().acceptAllPending();
    QMessageBox::information(this, QString::fromUtf8("提示"),
                             QString::fromUtf8("已接受 %1 项").arg(n));
    refreshFromCenter();
}

void PendingDialog::onRejectAll()
{
    const int n = PendingCenter::instance().rejectAllPending();
    QMessageBox::information(this, QString::fromUtf8("提示"),
                             QString::fromUtf8("已拒绝 %1 项").arg(n));
    refreshFromCenter();
}

void PendingDialog::onIgnoreAll()
{
    const int n = PendingCenter::instance().ignoreAllPending();
    QMessageBox::information(this, QString::fromUtf8("提示"),
                             QString::fromUtf8("已忽略 %1 项").arg(n));
    refreshFromCenter();
}

void PendingDialog::onClearDecided()
{
    const QMessageBox::StandardButton ret = QMessageBox::question(
        this,
        QString::fromUtf8("确认"),
        QString::fromUtf8("确定要清除所有已决定的项吗？（待确认项不会受影响）"),
        QMessageBox::Yes | QMessageBox::No);

    if (ret != QMessageBox::Yes) return;

    PendingCenter::instance().clearDecided();
    refreshFromCenter();
}

} // namespace analyze