#include "pending_center.h"

#include <QMutexLocker>

namespace analyze {

// ============================================================
// Singleton
// ============================================================
PendingCenter &PendingCenter::instance()
{
    static PendingCenter s_instance;
    return s_instance;
}

// ============================================================
// Add
// ============================================================
int PendingCenter::addItem(const PendingItem &item)
{
    QMutexLocker locker(&m_mutex);

    PendingItem copy = item;
    copy.id = m_nextId++;
    if (!copy.createdAt.isValid()) {
        copy.createdAt = QDateTime::currentDateTime();
    }
    m_items.append(copy);
    return copy.id;
}

QList<int> PendingCenter::addItems(const QList<PendingItem> &items)
{
    QMutexLocker locker(&m_mutex);

    QList<int> ids;
    ids.reserve(items.size());
    for (const PendingItem &it : items) {
        PendingItem copy = it;
        copy.id = m_nextId++;
        if (!copy.createdAt.isValid()) {
            copy.createdAt = QDateTime::currentDateTime();
        }
        m_items.append(copy);
        ids.append(copy.id);
    }
    return ids;
}

// ============================================================
// Query
// ============================================================
QList<PendingItem> PendingCenter::allItems() const
{
    QMutexLocker locker(&m_mutex);
    return m_items;
}

QList<PendingItem> PendingCenter::pendingItems() const
{
    QMutexLocker locker(&m_mutex);

    QList<PendingItem> result;
    for (const PendingItem &it : m_items) {
        if (it.decision == PendingDecision::Pending) {
            result.append(it);
        }
    }
    return result;
}

PendingItem PendingCenter::itemById(int id) const
{
    QMutexLocker locker(&m_mutex);

    for (const PendingItem &it : m_items) {
        if (it.id == id) {
            return it;
        }
    }
    return PendingItem();
}

int PendingCenter::count() const
{
    QMutexLocker locker(&m_mutex);
    return m_items.size();
}

int PendingCenter::pendingCount() const
{
    QMutexLocker locker(&m_mutex);

    int n = 0;
    for (const PendingItem &it : m_items) {
        if (it.decision == PendingDecision::Pending) {
            ++n;
        }
    }
    return n;
}

// ============================================================
// Decide
// ============================================================
bool PendingCenter::setDecision(int id, PendingDecision decision)
{
    QMutexLocker locker(&m_mutex);

    for (PendingItem &it : m_items) {
        if (it.id == id) {
            it.decision = decision;
            return true;
        }
    }
    return false;
}

bool PendingCenter::setDecisionByIds(const QList<int> &ids,
                                      PendingDecision decision)
{
    QMutexLocker locker(&m_mutex);

    bool allFound = true;
    for (int id : ids) {
        bool found = false;
        for (PendingItem &it : m_items) {
            if (it.id == id) {
                it.decision = decision;
                found = true;
                break;
            }
        }
        if (!found) allFound = false;
    }
    return allFound;
}

int PendingCenter::acceptAllPending()
{
    QMutexLocker locker(&m_mutex);

    int n = 0;
    for (PendingItem &it : m_items) {
        if (it.decision == PendingDecision::Pending) {
            it.decision = PendingDecision::Accepted;
            ++n;
        }
    }
    return n;
}

int PendingCenter::rejectAllPending()
{
    QMutexLocker locker(&m_mutex);

    int n = 0;
    for (PendingItem &it : m_items) {
        if (it.decision == PendingDecision::Pending) {
            it.decision = PendingDecision::Rejected;
            ++n;
        }
    }
    return n;
}

int PendingCenter::ignoreAllPending()
{
    QMutexLocker locker(&m_mutex);

    int n = 0;
    for (PendingItem &it : m_items) {
        if (it.decision == PendingDecision::Pending) {
            it.decision = PendingDecision::Ignored;
            ++n;
        }
    }
    return n;
}

// ============================================================
// Clear
// ============================================================
void PendingCenter::clear()
{
    QMutexLocker locker(&m_mutex);
    m_items.clear();
    m_nextId = 1;
}

void PendingCenter::clearDecided()
{
    QMutexLocker locker(&m_mutex);

    QList<PendingItem> keep;
    for (const PendingItem &it : m_items) {
        if (it.decision == PendingDecision::Pending) {
            keep.append(it);
        }
    }
    m_items = keep;
}

} // namespace analyze