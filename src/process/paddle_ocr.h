#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <opencv2/core.hpp>

namespace process {

// ============================================================
// PaddleOCR 识别结果
// ============================================================
struct PaddleOcrBox
{
    cv::Rect rect;         // 边界框（x, y, w, h）
    QString  text;         // 识别文本
    double   confidence;   // 置信度（0~1）
};

struct PaddleOcrResult
{
    bool ok = false;
    QString text;          // 所有块拼接的文本
    int     number = -1;   // 提取出来的数字
    double  confidence = 0.0;
    QList<PaddleOcrBox> boxes;   // 所有识别块
    QString errorMsg;
};

// ============================================================
// PaddleOCR-json 封装
//   - 通过 QProcess 启动 PaddleOCR-json.exe
//   - 使用 Base64 传图（避免中文路径问题）
//   - 解析 JSON 结果
// ============================================================
class PaddleOcr : public QObject
{
    Q_OBJECT

public:
    explicit PaddleOcr(QObject *parent = nullptr);
    ~PaddleOcr() override;

    // 启动引擎（指定 PaddleOCR-json.exe 路径）
    bool start(const QString &exePath);

    // 是否已就绪
    bool isReady() const { return m_ready; }

    // 识别一张 OpenCV Mat（BGR / 灰度都可以）
    PaddleOcrResult recognize(const cv::Mat &image);

    // 停止引擎
    void stop();

private:
    QProcess *m_process = nullptr;
    QString   m_exePath;
    bool      m_ready = false;

    // 等待引擎就绪
    bool waitForReady(int timeoutMs = 15000);

    // 发送 Base64 图片
    bool sendImage(const QByteArray &base64Image);

    // 读取一行 JSON 响应
    QByteArray readOneJsonLine(int timeoutMs);

    // 解析 JSON
    PaddleOcrResult parseResult(const QByteArray &jsonLine);
};

} // namespace process