#pragma once

#include <QString>
#include <QList>

namespace batch {

// ============================================================
// 处理步骤类型
// ============================================================
enum class StepType
{
    Deskew = 0,        // 自动扶正
    BlackEdge,         // 黑边去除
    Denoise,           // 污点去除（含印章保护）
    Enhance,           // 文字加深
    ErrPage,           // 错误页码处理
    Background,        // 底色处理
    ColorLine          // 彩色细线清除
};

// ============================================================
// 一个处理步骤
// ============================================================
struct StepItem
{
    StepType type = StepType::Deskew;
    bool enabled = true;
    int order = 0;          // 越小越先执行
};

// ============================================================
// 步骤名称
// ============================================================
inline QString stepName(StepType t)
{
    switch (t) {
    case StepType::Deskew:      return QString::fromUtf8("自动扶正");
    case StepType::BlackEdge:   return QString::fromUtf8("黑边去除");
    case StepType::Denoise:     return QString::fromUtf8("污点去除");
    case StepType::Enhance:     return QString::fromUtf8("文字加深");
    case StepType::ErrPage:     return QString::fromUtf8("错误页码处理");
    case StepType::Background:  return QString::fromUtf8("底色处理");
    case StepType::ColorLine:   return QString::fromUtf8("彩色细线清除");
    default:                    return QString::fromUtf8("未知");
    }
}

// ============================================================
// 步骤说明（用于 UI 提示）
// ============================================================
inline QString stepDescription(StepType t)
{
    switch (t) {
    case StepType::Deskew:
        return QString::fromUtf8("自动校正倾斜角度");
    case StepType::BlackEdge:
        return QString::fromUtf8("去除扫描产生的黑色边缘");
    case StepType::Denoise:
        return QString::fromUtf8("去除污点霉斑，自动保护印章和签名");
    case StepType::Enhance:
        return QString::fromUtf8("加深浅色文字（文字已够黑时自动跳过）");
    case StepType::ErrPage:
        return QString::fromUtf8("检测并清除错误页码（基于文件名判断）");
    case StepType::Background:
        return QString::fromUtf8("白化泛黄泛灰的纸张底色，保护彩色内容");
    case StepType::ColorLine:
        return QString::fromUtf8("检测并清除彩色故障细线");
    default:
        return QString();
    }
}

// ============================================================
// 默认步骤列表（全部启用，按推荐顺序）
// ============================================================
inline QList<StepItem> defaultSteps()
{
    QList<StepItem> steps;

    auto add = [&](StepType t, int order) {
        StepItem s;
        s.type = t;
        s.enabled = true;
        s.order = order;
        steps.append(s);
    };

    add(StepType::Deskew,      0);
    add(StepType::BlackEdge,   1);
    add(StepType::Denoise,     2);
    add(StepType::Enhance,     3);
    add(StepType::ErrPage,     4);
    add(StepType::Background,  5);
    add(StepType::ColorLine,   6);

    return steps;
}

} // namespace batch