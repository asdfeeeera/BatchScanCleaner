#pragma once

#include <QString>
#include <QDateTime>
#include <opencv2/core.hpp>

namespace analyze {

// ============================================================
// Pending item type
// ============================================================
enum class PendingType
{
    Unknown = 0,
    WrongPageNumber,     // detected page number does not match filename
    ColorLineDetected,   // colored thin line detected
    SpotSuspected,       // suspected spot
    SignatureOverlap,    // signature area needs confirmation
    Other
};

// ============================================================
// Suggested action
// ============================================================
enum class PendingAction
{
    Unknown = 0,
    Keep,                // keep the item in image
    Remove,              // remove the item from image
    ManualReview         // needs manual check
};

// ============================================================
// User decision
// ============================================================
enum class PendingDecision
{
    Pending = 0,         // not decided yet
    Accepted,            // user accepted suggested action
    Rejected,            // user rejected suggested action
    Ignored              // user ignored
};

// ============================================================
// One pending item
// ============================================================
struct PendingItem
{
    int id = -1;
    PendingType type = PendingType::Unknown;
    PendingAction suggestedAction = PendingAction::Unknown;
    PendingDecision decision = PendingDecision::Pending;

    QString sourceImagePath;   // full path of the image
    cv::Rect boundingBox;      // location inside image (for overlay)
    cv::Mat thumbnail;         // small preview (BGR or GRAY)
    QString reason;            // human-readable reason
    QString detail;            // extra detail (e.g. recognized number)
    double confidence = 0.0;   // 0.0 - 1.0
    QDateTime createdAt;       // when added

    // Helper: short text of type
    static QString typeToString(PendingType t)
    {
        switch (t) {
        case PendingType::WrongPageNumber: return QString::fromUtf8("错误页码");
        case PendingType::ColorLineDetected: return QString::fromUtf8("彩色细线");
        case PendingType::SpotSuspected: return QString::fromUtf8("疑似污点");
        case PendingType::SignatureOverlap: return QString::fromUtf8("签名重叠");
        case PendingType::Other: return QString::fromUtf8("其它");
        default: return QString::fromUtf8("未知");
        }
    }

    // Helper: short text of suggested action
    static QString actionToString(PendingAction a)
    {
        switch (a) {
        case PendingAction::Keep: return QString::fromUtf8("保留");
        case PendingAction::Remove: return QString::fromUtf8("删除");
        case PendingAction::ManualReview: return QString::fromUtf8("人工检查");
        default: return QString::fromUtf8("未知");
        }
    }

    // Helper: short text of decision
    static QString decisionToString(PendingDecision d)
    {
        switch (d) {
        case PendingDecision::Pending: return QString::fromUtf8("待确认");
        case PendingDecision::Accepted: return QString::fromUtf8("已接受");
        case PendingDecision::Rejected: return QString::fromUtf8("已拒绝");
        case PendingDecision::Ignored: return QString::fromUtf8("已忽略");
        default: return QString::fromUtf8("未知");
        }
    }
};

} // namespace analyze