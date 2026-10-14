#include "yolo_detector.h"

#include <QFileInfo>
#include <QDebug>
#include <QMutexLocker>

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace process {

// ============================================================
// 内部：letterbox（等比缩放 + 填充，YOLOv5 标准做法）
//   返回 640x640 的图，同时输出缩放比和 padding 偏移
// ============================================================
static cv::Mat letterboxImage(const cv::Mat &src,
                               int newSize,
                               const cv::Scalar &color,
                               float &outScale,
                               int &outPadW,
                               int &outPadH)
{
    const int w = src.cols;
    const int h = src.rows;

    outScale = std::min(static_cast<float>(newSize) / w,
                         static_cast<float>(newSize) / h);

    const int newW = static_cast<int>(std::round(w * outScale));
    const int newH = static_cast<int>(std::round(h * outScale));

    cv::Mat resized;
    if (newW != w || newH != h) {
        cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = src;
    }

    const int dw = newSize - newW;
    const int dh = newSize - newH;
    const int top = dh / 2;
    const int bottom = dh - top;
    const int left = dw / 2;
    const int right = dw - left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, color);

    outPadW = left;
    outPadH = top;
    return out;
}

// ============================================================
// 构造 / 析构
// ============================================================
YoloDetector::YoloDetector()
{
}

YoloDetector::~YoloDetector()
{
}

// ============================================================
// 加载 ONNX 模型
// ============================================================
bool YoloDetector::load(const QString &onnxPath,
                         float confThreshold,
                         float nmsThreshold)
{
    QMutexLocker locker(&m_mutex);

    if (!QFileInfo::exists(onnxPath)) {
        qWarning() << "[YoloDetector] model not found:" << onnxPath;
        return false;
    }

    try {
        m_net = cv::dnn::readNetFromONNX(onnxPath.toStdString());
    } catch (const cv::Exception &e) {
        qWarning() << "[YoloDetector] load failed:" << e.what();
        m_ready = false;
        return false;
    }

    if (m_net.empty()) {
        qWarning() << "[YoloDetector] net is empty";
        m_ready = false;
        return false;
    }

    // 纯 CPU 推理（避免 Windows 下 CUDA 后端依赖问题）
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    m_confThreshold = confThreshold;
    m_nmsThreshold  = nmsThreshold;
    m_ready = true;

    qDebug() << "[YoloDetector] model loaded:" << onnxPath;
    return true;
}

// ============================================================
// 检测
// ============================================================
std::vector<YoloBox> YoloDetector::detect(const cv::Mat &bgrImage)
{
    QMutexLocker locker(&m_mutex);

    std::vector<YoloBox> results;

    if (!m_ready) return results;
    if (bgrImage.empty()) return results;

    const int origW = bgrImage.cols;
    const int origH = bgrImage.rows;

    // 1. letterbox 到 640x640
    float scale = 1.0f;
    int padW = 0;
    int padH = 0;
    cv::Mat input = letterboxImage(bgrImage, m_inputSize,
                                    cv::Scalar(114, 114, 114),
                                    scale, padW, padH);

    // 2. 转 blob：BGR -> RGB，归一化到 0~1，HWC -> CHW
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob,
                            1.0 / 255.0,
                            cv::Size(m_inputSize, m_inputSize),
                            cv::Scalar(),
                            true,    // swapRB
                            false);  // crop

    // 3. 前向
    m_net.setInput(blob);
    cv::Mat output;
    try {
        output = m_net.forward();
    } catch (const cv::Exception &e) {
        qWarning() << "[YoloDetector] forward failed:" << e.what();
        return results;
    }

    // YOLOv5 输出：[1, 25200, 6]（单类别：x,y,w,h,obj,cls）
    if (output.dims != 3) {
        qWarning() << "[YoloDetector] unexpected output dims:" << output.dims;
        return results;
    }

    const int numBoxes = output.size[1];
    const int numAttrs = output.size[2];
    if (numAttrs < 6) {
        qWarning() << "[YoloDetector] unexpected attrs:" << numAttrs;
        return results;
    }

    // 输出数据是连续 float
    const float *data = reinterpret_cast<const float *>(output.data);

    std::vector<cv::Rect> boxes;
    std::vector<float>    confs;
    std::vector<int>      classIds;

    boxes.reserve(256);
    confs.reserve(256);
    classIds.reserve(256);

    for (int i = 0; i < numBoxes; ++i) {
        const float *p = data + static_cast<size_t>(i) * numAttrs;

        const float cx = p[0];
        const float cy = p[1];
        const float bw = p[2];
        const float bh = p[3];
        const float objConf = p[4];
        const float clsConf = p[5];

        const float conf = objConf * clsConf;
        if (conf < m_confThreshold) continue;

        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

        // 去掉 padding，还原到原图坐标
        if (scale <= 0.0f) continue;
        x1 = (x1 - padW) / scale;
        y1 = (y1 - padH) / scale;
        x2 = (x2 - padW) / scale;
        y2 = (y2 - padH) / scale;

        // 裁剪到原图范围
        x1 = std::max(0.0f, std::min(static_cast<float>(origW), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(origH), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(origW), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(origH), y2));

        const int ix1 = static_cast<int>(std::round(x1));
        const int iy1 = static_cast<int>(std::round(y1));
        const int ix2 = static_cast<int>(std::round(x2));
        const int iy2 = static_cast<int>(std::round(y2));

        if (ix2 <= ix1 || iy2 <= iy1) continue;

        boxes.push_back(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1));
        confs.push_back(conf);
        classIds.push_back(0);
    }

    if (boxes.empty()) return results;

    // 4. NMS 去重
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs,
                       m_confThreshold, m_nmsThreshold,
                       indices);

    for (int idx : indices) {
        YoloBox box;
        box.rect       = boxes[idx];
        box.confidence = confs[idx];
        box.classId    = classIds[idx];
        results.push_back(box);
    }

    return results;
}

} // namespace process