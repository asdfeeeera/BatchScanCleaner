#include "blackedge.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QFileInfo>
#include <QDir>
#include <QDateTime>
#include <QDebug>
#include <vector>
#include <algorithm>

namespace process {

BlackEdgeResult BlackEdge::removeBlackEdge(const cv::Mat &src, const BlackEdgeOptions &options)
{
    BlackEdgeResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;

    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src.clone();

    // 1. 提取印章掩膜（严格只认红章，绝不认错）
    cv::Mat colorMask = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // ★ 核心修正：H 必须在红色范围内（0-10 或 170-180），S 饱和度必须够高（>40），V 亮度不能太暗（>50）
        cv::Mat maskRed1, maskRed2;
        cv::inRange(hsv, cv::Scalar(0, 40, 50), cv::Scalar(10, 255, 255), maskRed1);
        cv::inRange(hsv, cv::Scalar(170, 40, 50), cv::Scalar(180, 255, 255), maskRed2);
        cv::bitwise_or(maskRed1, maskRed2, colorMask);
    }
    
    // 如果批处理外部传了掩膜，也一并加上，但前提是外部掩膜也是准的
    if (!options.protectMask.empty()) {
        cv::bitwise_or(colorMask, options.protectMask, colorMask);
    }

    // 稍微膨胀红章保护圈（12像素即可，防止边缘误切）
    if (cv::countNonZero(colorMask) > 0) {
        cv::dilate(colorMask, colorMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(12, 12)));
    }

    // 2. 自适应二值化抓暗区（发灰的阴影也能抓）
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 15);

    // 3. 大核开运算：抹除所有文字和细线
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::Mat opened;
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, openKernel);

    // 4. 大核闭运算：把反光断裂的黑边连接成大片
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
    cv::morphologyEx(opened, opened, cv::MORPH_CLOSE, closeKernel);

    // 5. 寻找大块连通域
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat blackMask = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        bool touchesEdge = (r.x <= 40) || (r.y <= 40) || (r.x + r.width >= W - 40) || (r.y + r.height >= H - 40);
        bool isBig = (r.width > 60 || r.height > 60);
        bool isWholePage = (r.width >= W - 20 && r.height >= H - 20);

        if (touchesEdge && isBig && !isWholePage) {
            cv::drawContours(blackMask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 6. 黑边掩膜轻度膨胀，吃掉边缘过渡带
    cv::dilate(blackMask, blackMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    // 7. 扣除印章！
    cv::Mat finalMask;
    cv::bitwise_and(blackMask, ~colorMask, finalMask);

    // ============================================================
    // 生成 Debug 图
    // ============================================================
    try {
        cv::Mat debugImg = src.clone();
        if (debugImg.channels() == 1) cv::cvtColor(debugImg, debugImg, cv::COLOR_GRAY2BGR);
        
        // 红色区域 = 被判定为黑边要涂白的
        if (debugImg.channels() == 3) debugImg.setTo(cv::Scalar(0, 0, 255), finalMask);
        else debugImg.setTo(cv::Scalar(0, 0, 255, 255), finalMask);
        
        // 绿色区域 = 被判定为印章保护起来的
        if (debugImg.channels() == 3) debugImg.setTo(cv::Scalar(0, 255, 0), colorMask);
        else debugImg.setTo(cv::Scalar(0, 255, 0, 255), colorMask);

        QString debugPath = QDir::currentPath() + "/debug_mask_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".jpg";
        cv::imwrite(debugPath.toLocal8Bit().toStdString(), debugImg);
        qDebug() << "【调试信息】已生成调试图:" << debugPath;
    } catch (...) {}
    // ============================================================

    // 8. 执行涂白
    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (src.channels() == 4) white = cv::Scalar(255, 255, 255, 255);
    else if (src.channels() == 3) white = cv::Scalar(255, 255, 255);
    else white = cv::Scalar(255);

    dst.setTo(white, finalMask);

    result.image = dst;
    result.filledPixels = cv::countNonZero(finalMask);
    result.ok = true;
    return result;
}

} // namespace process