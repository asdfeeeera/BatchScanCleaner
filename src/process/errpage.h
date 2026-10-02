#pragma once

#include <opencv2/core.hpp>
#include <QString>
#include <vector>

namespace process {

struct PageNumberItem
{
    cv::Rect boundingBox;
    bool isCrossed = false;
    int recognizedNumber = -1;
    QString recognizedText;
    double confidence = 0.0;
};

struct ErrPageResult
{
    cv::Mat image;
    cv::Mat markedImage;
    std::vector<PageNumberItem> items;
    int correctPage = -1;
    int crossedRemoved = 0;
    int pendingCount = 0;
    bool ok = false;
    bool skipped = false;
};

struct ErrPageOptions
{
    // 左上角（小部分文件）
    bool detectTopLeft = true;
    double topLeftWidthRatio = 0.12;
    double topLeftHeightRatio = 0.08;

    // 右上角（大部分文件）
    bool detectTopRight = true;
    double topRightWidthRatio = 0.25;
    double topRightHeightRatio = 0.15;

    // 右下角（横向文件）
    bool detectBottomRight = true;
    double bottomRightWidthRatio = 0.25;
    double bottomRightHeightRatio = 0.15;

    // 左下角（可选，默认关闭）
    bool detectBottomLeft = false;
    double bottomLeftWidthRatio = 0.12;
    double bottomLeftHeightRatio = 0.08;

    // 数字区域最小/最大尺寸
    int minDigitHeight = 20;
    int maxDigitHeight = 100;
    int minDigitWidth = 10;
    int maxDigitWidth = 100;

    double crossLineRatio = 0.5;
    bool fillWhite = true;
    QString tesseractPath;
};

class ErrPage
{
public:
    static ErrPageResult process(const cv::Mat &src,
                                  const QString &sourcePath,
                                  const ErrPageOptions &options = ErrPageOptions());

    static int parseCorrectPage(const QString &sourcePath);

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