#pragma once

#include <QDialog>
#include "pending_item.h"

class QListWidget;
class QListWidgetItem;
class QLabel;
class QTextEdit;
class QPushButton;

namespace analyze {

// ============================================================
// PendingDialog
// UI dialog to review and decide all pending items.
// Left: list of items. Right: preview + details.
// Bottom: action buttons.
// ============================================================
class PendingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PendingDialog(QWidget *parent = nullptr);
    ~PendingDialog() override;

    // Reload items from PendingCenter
    void refreshFromCenter();

private slots:
    void onItemSelectionChanged();
    void onAcceptCurrent();
    void onRejectCurrent();
    void onIgnoreCurrent();
    void onAcceptAll();
    void onRejectAll();
    void onIgnoreAll();
    void onClearDecided();

private:
    void setupUi();
    void updateDetailPanel();
    void updateButtonsState();
    void rebuildList();
    QString formatListItemText(const PendingItem &it) const;

    QListWidget *m_listWidget = nullptr;
    QLabel *m_thumbnailLabel = nullptr;
    QTextEdit *m_detailEdit = nullptr;
    QLabel *m_statsLabel = nullptr;

    QPushButton *m_acceptBtn = nullptr;
    QPushButton *m_rejectBtn = nullptr;
    QPushButton *m_ignoreBtn = nullptr;
    QPushButton *m_acceptAllBtn = nullptr;
    QPushButton *m_rejectAllBtn = nullptr;
    QPushButton *m_ignoreAllBtn = nullptr;
    QPushButton *m_clearDecidedBtn = nullptr;
    QPushButton *m_closeBtn = nullptr;
};

} // namespace analyze