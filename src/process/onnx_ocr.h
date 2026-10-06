#pragma once

#include <opencv2/core.hpp>
#include <QString>
#include <QStringList>

namespace process {

// ============================================================
// ONNX 数字识别器（替代 Tesseract）
//   - 加载 best_ocr.onnx
//   - 3 位数字分类（3 个头，每个 10 类）
//   - 线程安全（懒加载 + 只读）
// ============================================================
class OnnxOcr
{
public:
    struct Result
    {
        bool ok = false;            // 是否成功识别
        int  number = -1;           // 识别结果（3 位数字，如 063）
        double confidence = 0.0;    // 最低置信度（三个头中最小的）
        QString text;               // 3 位字符串（如 "063"）
    };

    // 单例：进程内只有一个 ONNX session
    static OnnxOcr &instance();

    // 设置 ONNX 模型路径（首次调用时生效；之后修改路径需重启程序）
    // 不设置则默认在 exe 同级目录找 best_ocr.onnx
    void setModelPath(const QString &path);

    // 是否已成功加载模型
    bool isReady() const;

    // 识别单张数字小图
    //   digitImage: 任意尺寸的 BGR 或灰度小图
    //   返回：识别结果
    Result recognize(const cv::Mat &digitImage);

    // 最小置信度阈值（低于此值返回 ok=false，让调用方走 fallback）
    void setMinConfidence(double v) { m_minConfidence = v; }
    double minConfidence() const { return m_minConfidence; }

private:
    OnnxOcr();
    ~OnnxOcr();
    OnnxOcr(const OnnxOcr &) = delete;
    OnnxOcr &operator=(const OnnxOcr &) = delete;

    bool ensureLoaded();
    QString resolveModelPath() const;

    QString m_modelPath;
    bool    m_triedLoad = false;
    bool    m_ready = false;
    double  m_minConfidence = 0.5;   // 低于 50% 视为不可信

    struct Impl;
    Impl *m_impl = nullptr;
};

} // namespace process