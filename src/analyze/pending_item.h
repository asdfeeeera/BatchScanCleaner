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
    WrongPageNumber,     // 错误页码
    ColorLineDetected,   // 彩色细线
    SpotSuspected,       // 疑似污点
    SignatureOverlap,    // 签名重叠
    YellowBlob,          // ★ 新增：黄色污渍
    Other
};

// ============================================================
// Suggested action
// ============================================================
enum class PendingAction
{
    Unknown = 0,
    Keep,                // 保留
    Remove,              // 删除
    ManualReview         // 人工检查
};

// ============================================================
// User decision
// ============================================================
enum class PendingDecision
{
    Pending = 0,
    Accepted,
    Rejected,
    Ignored
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

    QString sourceImagePath;
    QString fileName;
    cv::Rect boundingBox;
    cv::Mat thumbnail;
    QString reason;
    QString detail;
    double confidence = 0.0;
    QDateTime createdAt;

    static QString typeToString(PendingType t)
    {
        switch (t) {
        case PendingType::WrongPageNumber: return QString::fromUtf8("错误页码");
        case PendingType::ColorLineDetected: return QString::fromUtf8("彩色细线");
        case PendingType::SpotSuspected: return QString::fromUtf8("疑似污点");
        case PendingType::SignatureOverlap: return QString::fromUtf8("签名重叠");
        case PendingType::YellowBlob: return QString::fromUtf8("黄色污渍");
        case PendingType::Other: return QString::fromUtf8("其它");
        default: return QString::fromUtf8("未知");
        }
    }

    static QString actionToString(PendingAction a)
    {
        switch (a) {
        case PendingAction::Keep: return QString::fromUtf8("保留");
        case PendingAction::Remove: return QString::fromUtf8("删除");
        case PendingAction::ManualReview: return QString::fromUtf8("人工检查");
        default: return QString::fromUtf8("未知");
        }
    }

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