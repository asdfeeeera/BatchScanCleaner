#include "paddle_ocr.h"

#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <QElapsedTimer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QJsonParseError>

#include <opencv2/imgcodecs.hpp>

#include <vector>

namespace process {

// ============================================================
// 构造 / 析构
// ============================================================
PaddleOcr::PaddleOcr(QObject *parent)
    : QObject(parent)
{
}

PaddleOcr::~PaddleOcr()
{
    stop();
}

// ============================================================
// 启动引擎
//   - 使用匿名管道模式（不带命令行参数启动 exe）
//   - 启动后等待 "OCR init completed."
// ============================================================
bool PaddleOcr::start(const QString &exePath)
{
    if (m_ready) return true;
    if (m_process) stop();

    m_exePath = exePath;

    if (!QFileInfo::exists(exePath)) {
        qWarning() << "[PaddleOcr] exe not found:" << exePath;
        return false;
    }

    m_process = new QProcess(this);
    m_process->setWorkingDirectory(QFileInfo(exePath).absolutePath());

    // stdout / stderr 合并，避免 init 日志在 stderr 时读不到
    m_process->setProcessChannelMode(QProcess::MergedChannels);

    // 无参数启动 → 匿名管道模式
    m_process->start(exePath, QStringList());

    if (!m_process->waitForStarted(5000)) {
        qWarning() << "[PaddleOcr] start failed:" << m_process->errorString();
        delete m_process;
        m_process = nullptr;
        return false;
    }

    if (!waitForReady(20000)) {
        qWarning() << "[PaddleOcr] init failed, stopping";
        stop();
        return false;
    }

    m_ready = true;
    qDebug() << "[PaddleOcr] engine ready:" << exePath;
    return true;
}

// ============================================================
// 等待引擎就绪
// ============================================================
bool PaddleOcr::waitForReady(int timeoutMs)
{
    QElapsedTimer timer;
    timer.start();

    while (timer.elapsed() < timeoutMs) {
        if (!m_process || m_process->state() != QProcess::Running) {
            qWarning() << "[PaddleOcr] process died during init";
            return false;
        }
        if (!m_process->waitForReadyRead(200)) continue;

        while (m_process->canReadLine()) {
            const QByteArray line = m_process->readLine().trimmed();
            if (line.isEmpty()) continue;
            if (line.contains("OCR init completed.")) {
                return true;
            }
        }
    }

    qWarning() << "[PaddleOcr] init timeout";
    return false;
}

// ============================================================
// 识别一张图（Base64 传图，避开中文路径问题）
// ============================================================
PaddleOcrResult PaddleOcr::recognize(const cv::Mat &image)
{
    PaddleOcrResult result;

    if (!m_ready || !m_process) {
        result.errorMsg = QString::fromUtf8("引擎未就绪");
        return result;
    }
    if (image.empty()) {
        result.errorMsg = QString::fromUtf8("输入图像为空");
        return result;
    }

    // 1. 编码为 JPG
    std::vector<uchar> buf;
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(95);

    if (!cv::imencode(".jpg", image, buf, params)) {
        result.errorMsg = QString::fromUtf8("图片编码失败");
        return result;
    }

    QByteArray jpgBytes(reinterpret_cast<const char*>(buf.data()),
                        static_cast<int>(buf.size()));
    QByteArray base64 = jpgBytes.toBase64();

    // 2. 发送
    if (!sendImage(base64)) {
        result.errorMsg = QString::fromUtf8("发送图片失败");
        return result;
    }

    // 3. 读取响应
    QByteArray jsonLine = readOneJsonLine(30000);
    if (jsonLine.isEmpty()) {
        result.errorMsg = QString::fromUtf8("读取 OCR 响应超时");
        return result;
    }

    // 4. 解析
    return parseResult(jsonLine);
}

// ============================================================
// 发送 Base64 图片
// ============================================================
bool PaddleOcr::sendImage(const QByteArray &base64Image)
{
    if (!m_process) return false;

    QJsonObject cmd;
    cmd.insert(QStringLiteral("image_base64"),
               QString::fromLatin1(base64Image));

    QJsonDocument doc(cmd);
    QByteArray json = doc.toJson(QJsonDocument::Compact);
    json.append('\n');

    const qint64 written = m_process->write(json);
    if (written < 0) return false;

    return m_process->waitForBytesWritten(5000);
}

// ============================================================
// 读取一行 JSON 响应
//   跳过日志行（非 { 开头）
// ============================================================
QByteArray PaddleOcr::readOneJsonLine(int timeoutMs)
{
    QElapsedTimer timer;
    timer.start();

    while (timer.elapsed() < timeoutMs) {
        if (!m_process || m_process->state() != QProcess::Running) {
            return QByteArray();
        }
        if (!m_process->waitForReadyRead(200)) continue;

        while (m_process->canReadLine()) {
            QByteArray line = m_process->readLine().trimmed();
            if (line.isEmpty()) continue;
            if (line.startsWith('{')) {
                return line;
            }
            // 其它行是日志，忽略
        }
    }
    return QByteArray();
}

// ============================================================
// 解析 JSON
//   ★ PaddleOCR-json v1.4.1 的置信度字段名是 "score"，
//     这里同时兼容 "score" 与 "confidence"，优先用 "score"
// ============================================================
PaddleOcrResult PaddleOcr::parseResult(const QByteArray &jsonLine)
{
    PaddleOcrResult result;

    QJsonParseError err;
    const QJsonDocument doc = QJsonDocument::fromJson(jsonLine, &err);
    if (err.error != QJsonParseError::NoError) {
        result.errorMsg = QString::fromUtf8("JSON 解析失败：%1")
                              .arg(err.errorString());
        return result;
    }

    if (!doc.isObject()) {
        result.errorMsg = QString::fromUtf8("JSON 格式错误");
        return result;
    }

    const QJsonObject root = doc.object();
    const int code = root.value(QStringLiteral("code")).toInt();

    if (code != 100) {
        result.errorMsg = QString::fromUtf8("OCR 返回错误：code=%1 data=%2")
                              .arg(code)
                              .arg(root.value(QStringLiteral("data")).toString());
        return result;
    }

    const QJsonArray dataArr = root.value(QStringLiteral("data")).toArray();

    result.ok = true;
    result.confidence = 1.0;

    QString allText;

    for (const QJsonValue &v : dataArr) {
        const QJsonObject obj = v.toObject();

        PaddleOcrBox box;
        box.text = obj.value(QStringLiteral("text")).toString();

        // ★ 关键修复：PaddleOCR-json 用的是 "score"，不是 "confidence"
        double conf = obj.value(QStringLiteral("score")).toDouble();
        if (conf <= 0.0) {
            // 兼容旧版本 / 其他命名
            conf = obj.value(QStringLiteral("confidence")).toDouble();
        }
        box.confidence = conf;

        const QJsonArray boxArr = obj.value(QStringLiteral("box")).toArray();
        if (boxArr.size() >= 4) {
            // 4 个点：[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            const QJsonArray p0 = boxArr.at(0).toArray();
            const QJsonArray p2 = boxArr.at(2).toArray();
            if (p0.size() >= 2 && p2.size() >= 2) {
                const int x1 = static_cast<int>(p0.at(0).toDouble());
                const int y1 = static_cast<int>(p0.at(1).toDouble());
                const int x2 = static_cast<int>(p2.at(0).toDouble());
                const int y2 = static_cast<int>(p2.at(1).toDouble());
                box.rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            }
        }

        result.boxes.append(box);
        allText += box.text;

        // 取最小置信度作为整体置信度（只要有一个块不确定就保守处理）
        if (box.confidence > 0.0 && box.confidence < result.confidence) {
            result.confidence = box.confidence;
        }
    }

    result.text = allText;

    // 从文本提取数字
    QString digits;
    for (const QChar &c : allText) {
        if (c.isDigit()) digits += c;
    }
    if (!digits.isEmpty()) {
        bool ok = false;
        const int n = digits.toInt(&ok);
        result.number = ok ? n : -1;
    }

    return result;
}

// ============================================================
// 停止引擎
// ============================================================
void PaddleOcr::stop()
{
    if (m_process) {
        if (m_process->state() == QProcess::Running) {
            m_process->closeWriteChannel();
            if (!m_process->waitForFinished(2000)) {
                m_process->kill();
                m_process->waitForFinished(1000);
            }
        }
        delete m_process;
        m_process = nullptr;
    }
    m_ready = false;
}

} // namespace process