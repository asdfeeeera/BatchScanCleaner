#pragma once

#include <opencv2/core.hpp>
#include <QString>
#include <vector>

namespace process {

// 单个页码项
struct PageNumberItem
{
    cv::Rect boundingBox;     // 外接矩形
    bool isCrossed = false;   // 是否有划线/打叉
    int recognizedNumber = -1; // OCR 识别出的数字（-1=未识别）
    QString recognizedText;   // OCR 原始文本
    double confidence = 0.0;  // OCR 置信度 0-1
};

struct ErrPageResult
{
    cv::Mat image;                                // 处理后的图
    cv::Mat markedImage;                          // 标记后的图
    std::vector<PageNumberItem> items;            // 检测到的所有页码
    int correctPage = -1;                         // 从文件名解析的正确页码
    int crossedRemoved = 0;                       // 自动删除的划线错误页码数
    int pendingCount = 0;                         // 待确认数量
    bool ok = false;
    bool skipped = false;
};

struct ErrPageOptions
{
    // 页码区域检测范围（占页面的比例）
    double regionWidthRatio = 0.25;
    double regionHeightRatio = 0.15;

    // 是否检测左上角
    bool detectTopLeft = true;
    bool detectTopRight = true;

    // 数字区域最小/最大尺寸（像素）
    int minDigitHeight = 15;
    int maxDigitHeight = 120;
    int minDigitWidth = 8;
    int maxDigitWidth = 120;

    // 划线判定：横线穿过数字区域的比例
    double crossLineRatio = 0.6;

    // 删除填充色
    bool fillWhite = true;

    // Tesseract 可执行文件路径（默认 exe 同目录下的 tesseract\tesseract.exe）
    QString tesseractPath;
};

class ErrPage
{
public:
    // 处理错误页码
    static ErrPageResult process(const cv::Mat &src,
                                  const QString &sourcePath,
                                  const ErrPageOptions &options = ErrPageOptions());

    // 从文件名解析正确页码
    static int parseCorrectPage(const QString &sourcePath);

    // 用 Tesseract OCR 识别数字块
    static int recognizeWithTesseract(const cv::Mat &digitImage,
                                       const QString &tesseractPath,
                                       QString &outText,
                                       double &outConfidence);

private:
    static void detectDigitsInRegion(const cv::Mat &gray,
                                      const cv::Rect &region,
                                      const ErrPageOptions &options,
                                      std::vector<PageNumberItem> &outItems);

    static bool detectCrossLine(const cv::Mat &gray,
                                 const cv::Rect &digitBox,
                                 double crossLineRatio);
};

} // namespace process