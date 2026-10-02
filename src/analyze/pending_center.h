#pragma once

#include "pending_item.h"

#include <QList>
#include <QMutex>

namespace analyze {

// ============================================================
// PendingCenter
// Global singleton that collects pending items from all
// processing modules, and lets the UI query / decide them.
// ============================================================
class PendingCenter
{
public:
    // Singleton access
    static PendingCenter &instance();

    // ---- Add ----
    // Add one item, returns assigned id (>= 1)
    int addItem(const PendingItem &item);

    // Add many items, returns list of assigned ids
    QList<int> addItems(const QList<PendingItem> &items);

    // ---- Query ----
    QList<PendingItem> allItems() const;
    QList<PendingItem> pendingItems() const;    // decision == Pending
    PendingItem itemById(int id) const;
    int count() const;
    int pendingCount() const;

    // ---- Decide ----
    bool setDecision(int id, PendingDecision decision);
    bool setDecisionByIds(const QList<int> &ids, PendingDecision decision);

    // Batch: mark all pending items as Accepted / Rejected / Ignored
    int acceptAllPending();
    int rejectAllPending();
    int ignoreAllPending();

    // ---- Clear ----
    void clear();
    // Clear only decided items (keep pending)
    void clearDecided();

private:
    PendingCenter() = default;
    ~PendingCenter() = default;
    PendingCenter(const PendingCenter &) = delete;
    PendingCenter &operator=(const PendingCenter &) = delete;

    mutable QMutex m_mutex;
    QList<PendingItem> m_items;
    int m_nextId = 1;
};

} // namespace analyze