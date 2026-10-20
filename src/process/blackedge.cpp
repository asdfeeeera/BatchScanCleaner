#include "blackedge.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> // ★ 修复 imwrite 缺失
#include <QFileInfo>             // ★ 修复路径问题
#include <QDir>
#include <QDateTime>             // ★ 修复 currentDateTime 缺失
#include <QDebug>                // ★ 修复 QDebug 缺失
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

    // 1. 提取极度宽泛的彩色印章掩膜
    cv::Mat colorMask;
    if (!options.protectMask.empty()) {
        colorMask = options.protectMask.clone();
    }
    
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        cv::Mat colorMask2;
        cv::inRange(hsvCh[1], 15, 255, colorMask2);
        cv::Mat valMask;
        cv::inRange(hsvCh[2], 20, 255, valMask); 
        cv::bitwise_and(colorMask2, valMask, colorMask2);
        
        if (colorMask.empty()) colorMask = colorMask2;
        else cv::bitwise_or(colorMask, colorMask2, colorMask);
    }

    // ★ 印章膨胀保护 (先用 25 像素)
    cv::Mat colorDilated;
    if (!colorMask.empty()) {
        cv::dilate(colorMask, colorDilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25)));
    } else {
        colorDilated = cv::Mat::zeros(gray.size(), CV_8UC1);
    }

    // 2. 自适应二值化抓暗区
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 10);

    // 3. 形态学去除文字，连接黑边
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::Mat opened;
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, openKernel);

    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::morphologyEx(opened, opened, cv::MORPH_CLOSE, closeKernel);

    // 4. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat blackMask = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        bool touchesEdge = (r.x <= 40) || (r.y <= 40) || (r.x + r.width >= W - 40) || (r.y + r.height >= H - 40);
        bool isBig = (r.width > 80 || r.height > 80);
        bool isWholePage = (r.width >= W - 20 && r.height >= H - 20);

        if (touchesEdge && isBig && !isWholePage) {
            cv::drawContours(blackMask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 5. 黑边掩膜膨胀
    cv::dilate(blackMask, blackMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));

    // 6. 从黑边掩膜中扣除印章保护区
    cv::Mat finalMask;
    cv::bitwise_and(blackMask, ~colorDilated, finalMask);

    // ============================================================
    // ★ 核心改动：生成 Debug 图！
    // 红色 = 被判定为黑边并涂白的区域
    // 绿色 = 被判定为印章并保护起来的区域
    // 剩余 = 原图（文字、表格等）
    // ============================================================
    try {
        cv::Mat debugImg = src.clone();
        if (debugImg.channels() == 1) cv::cvtColor(debugImg, debugImg, cv::COLOR_GRAY2BGR);

        // 画黑边（红色）
        if (src.channels() == 3) debugImg.setTo(cv::Scalar(0, 0, 255), finalMask);
        else debugImg.setTo(cv::Scalar(0, 0, 255, 255), finalMask);

        // 画印章保护（绿色）
        if (src.channels() == 3) debugImg.setTo(cv::Scalar(0, 255, 0), colorDilated);
        else debugImg.setTo(cv::Scalar(0, 255, 0, 255), colorDilated);

        QString debugPath = QDir::currentPath() + "/debug_mask_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".jpg";
        cv::imwrite(debugPath.toLocal8Bit().toStdString(), debugImg);
        qDebug() << "【调试信息】已生成黑边/印章Mask调试图:" << debugPath;
    } catch (...) {}
    // ============================================================

    // 7. 执行纯涂白
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