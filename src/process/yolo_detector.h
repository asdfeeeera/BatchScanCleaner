#pragma once

#include <QString>
#include <QMutex>
#include <opencv2/core.hpp>
#include <vector>

namespace process {

// ============================================================
// YOLO 检测结果
// ============================================================
struct YoloBox
{
    cv::Rect rect;         // 原图坐标
    float    confidence;   // 置信度（0~1）
    int      classId;      // 类别 id（本模型只有 0 = page）
};

// ============================================================
// YOLOv5 ONNX 检测器
//   - 用 OpenCV DNN 加载
//   - 输入：BGR cv::Mat（任意尺寸）
//   - 输出：原图坐标系下的检测框列表
//   - 线程安全（内部有 QMutex）
// ============================================================
class YoloDetector
{
public:
    YoloDetector();
    ~YoloDetector();

    // 加载模型（onnx 路径）
    bool load(const QString &onnxPath,
              float confThreshold = 0.25f,
              float nmsThreshold  = 0.45f);

    // 是否已加载
    bool isReady() const { return m_ready; }

    // 检测一张图，返回所有目标框（原图坐标）
    std::vector<YoloBox> detect(const cv::Mat &bgrImage);

private:
    cv::dnn::Net m_net;
    bool         m_ready = false;
    float        m_confThreshold = 0.25f;
    float        m_nmsThreshold  = 0.45f;
    int          m_inputSize     = 640;   // YOLOv5s 默认

    mutable QMutex m_mutex;
};

} // namespace process