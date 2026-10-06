#include "onnx_ocr.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <QMutex>
#include <QMutexLocker>
#include <QDebug>

#include <cmath>
#include <exception>
#include <vector>

namespace process {

// ============================================================
// Impl：保存 ONNX Net 和锁
// ============================================================
struct OnnxOcr::Impl
{
    cv::dnn::Net net;
    QMutex       mutex;
};

// ============================================================
// 单例
// ============================================================
OnnxOcr &OnnxOcr::instance()
{
    static OnnxOcr s;
    return s;
}

OnnxOcr::OnnxOcr()
    : m_impl(new Impl)
{
}

OnnxOcr::~OnnxOcr()
{
    delete m_impl;
    m_impl = nullptr;
}

// ============================================================
// 路径与状态
// ============================================================
void OnnxOcr::setModelPath(const QString &path)
{
    m_modelPath = path;
    m_triedLoad = false;
    m_ready = false;
}

bool OnnxOcr::isReady() const
{
    return m_ready;
}

QString OnnxOcr::resolveModelPath() const
{
    if (!m_modelPath.trimmed().isEmpty()) {
        return m_modelPath;
    }
    // 默认：exe 同级目录下的 best_ocr.onnx
    return QCoreApplication::applicationDirPath()
           + QStringLiteral("/best_ocr.onnx");
}

// ============================================================
// 懒加载
// ============================================================
bool OnnxOcr::ensureLoaded()
{
    if (m_ready) return true;
    if (m_triedLoad) return false;   // 只尝试一次
    m_triedLoad = true;

    const QString path = resolveModelPath();
    if (!QFileInfo::exists(path)) {
        qWarning() << "[OnnxOcr] model file not found:" << path;
        return false;
    }

    try {
        m_impl->net = cv::dnn::readNetFromONNX(path.toStdString());
        m_impl->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_impl->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        m_ready = true;
        qDebug() << "[OnnxOcr] model loaded:" << path;
    } catch (const std::exception &e) {
        qWarning() << "[OnnxOcr] load failed:" << e.what();
        m_ready = false;
    } catch (...) {
        qWarning() << "[OnnxOcr] load failed: unknown exception";
        m_ready = false;
    }

    return m_ready;
}

// ============================================================
// 识别
// ============================================================
OnnxOcr::Result OnnxOcr::recognize(const cv::Mat &digitImage)
{
    Result r;

    if (digitImage.empty()) return r;
    if (!ensureLoaded()) return r;

    // ---------- 1. 转灰度 ----------
    cv::Mat gray;
    if (digitImage.channels() == 1) {
        gray = digitImage;
    } else if (digitImage.channels() == 4) {
        cv::cvtColor(digitImage, gray, cv::COLOR_BGRA2GRAY);
    } else {
        cv::cvtColor(digitImage, gray, cv::COLOR_BGR2GRAY);
    }

    // ---------- 2. 缩放到 128x48 ----------
    const int TARGET_W = 128;
    const int TARGET_H = 48;

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(TARGET_W, TARGET_H),
               0, 0, cv::INTER_AREA);

    // ---------- 3. 归一化到 [-1, 1] ----------
    //   f = src * (2/255) - 1
    cv::Mat f;
    resized.convertTo(f, CV_32F, 2.0 / 255.0, -1.0);

    // ---------- 4. 构造 blob（NCHW）----------
    cv::Mat blob = cv::dnn::blobFromImage(f);

    // ---------- 5. 推理 ----------
    std::vector<cv::Mat> outputs;

    {
        QMutexLocker locker(&m_impl->mutex);

        m_impl->net.setInput(blob);
        m_impl->net.forward(outputs,
            m_impl->net.getUnconnectedOutLayersNames());
    }

    if (outputs.size() < 3) {
        qWarning() << "[OnnxOcr] expected 3 outputs, got" << outputs.size();
        return r;
    }

    // ---------- 6. 解析 3 个 head ----------
    int    digits[3] = {0, 0, 0};
    double minConf = 1.0;

    for (int i = 0; i < 3; ++i) {
        const cv::Mat &out = outputs[i];
        if (out.total() < 10) return r;

        const float *p = out.ptr<float>(0);

        // softmax 分母
        float maxLogit = p[0];
        for (int j = 1; j < 10; ++j) {
            if (p[j] > maxLogit) maxLogit = p[j];
        }

        double sumExp = 0.0;
        for (int j = 0; j < 10; ++j) {
            sumExp += std::exp(static_cast<double>(p[j] - maxLogit));
        }

        int    bestIdx  = 0;
        double bestExp  = std::exp(static_cast<double>(p[0] - maxLogit));
        for (int j = 1; j < 10; ++j) {
            double e = std::exp(static_cast<double>(p[j] - maxLogit));
            if (e > bestExp) {
                bestExp = e;
                bestIdx = j;
            }
        }

        double conf = bestExp / sumExp;

        digits[i] = bestIdx;
        if (conf < minConf) minConf = conf;
    }

    // ---------- 7. 组装结果 ----------
    r.confidence = minConf;
    r.number     = digits[0] * 100 + digits[1] * 10 + digits[2];
    r.text       = QString("%1%2%3")
                       .arg(digits[0])
                       .arg(digits[1])
                       .arg(digits[2]);
    r.ok = (minConf >= m_minConfidence);

    return r;
}

} // namespace process