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

    // 1. 提取极度宽泛的印章掩膜（自建 + 外部传入）
    cv::Mat colorMask = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // 极宽阈值：饱和 > 15，亮度 > 20 就算印章
        cv::Mat maskSelf;
        cv::inRange(hsvCh[1], 15, 255, maskSelf);
        cv::Mat valMask;
        cv::inRange(hsvCh[2], 20, 255, valMask); 
        cv::bitwise_and(maskSelf, valMask, maskSelf);
        cv::bitwise_or(colorMask, maskSelf, colorMask);
    }
    
    if (!options.protectMask.empty()) {
        cv::bitwise_or(colorMask, options.protectMask, colorMask);
    }

    // ★ 印章疯狂外扩 30 像素！绝对护盾！
    if (!colorMask.empty()) {
        cv::dilate(colorMask, colorMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30)));
    }

    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (src.channels() == 4) white = cv::Scalar(255, 255, 255, 255);
    else if (src.channels() == 3) white = cv::Scalar(255, 255, 255);
    else white = cv::Scalar(255);

    int paintedCount = 0;

    auto setWhite = [&](int y, int x) {
        if (dst.channels() == 3) dst.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
        else if (dst.channels() == 4) dst.at<cv::Vec4b>(y, x) = cv::Vec4b(255, 255, 255, 255);
        else dst.at<uchar>(y, x) = 255;
        paintedCount++;
    };

    const double darkThr = 180.0; // 小于 180 全算黑边
    const int maxScanX = static_cast<int>(W * 0.35); // 允许向里深挖
    const int maxScanY = static_cast<int>(H * 0.35);
    const int maxGap = 50; // 容忍 50 像素反光断点

    // --- 1. 上边缘 ---
    for (int x = 0; x < W; ++x) {
        int cut = 0;
        int whiteRun = 0;
        for (int y = 0; y < maxScanY; ++y) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) {
                cut = y + 1; // 印章也计入涂白深度（为了越过它找后面的黑边），但后面会绕过涂白
                whiteRun = 0;
                continue;
            }
            uchar v = gray.at<uchar>(y, x);
            if (v < darkThr) {
                cut = y + 1;
                whiteRun = 0;
            } else if (v > 235) {
                whiteRun++;
                if (whiteRun > maxGap) break;
            } else {
                cut = y + 1;
                whiteRun = 0;
            }
        }
        for (int y = 0; y < cut; ++y) {
            if (colorMask.empty() || colorMask.at<uchar>(y, x) == 0) setWhite(y, x);
        }
    }

    // --- 2. 下边缘 ---
    for (int x = 0; x < W; ++x) {
        int cut = 0;
        int whiteRun = 0;
        for (int y = 0; y < maxScanY; ++y) {
            if (!colorMask.empty() && colorMask.at<uchar>(H - 1 - y, x) > 0) {
                cut = y + 1; whiteRun = 0; continue;
            }
            uchar v = gray.at<uchar>(H - 1 - y, x);
            if (v < darkThr) { cut = y + 1; whiteRun = 0; }
            else if (v > 235) { whiteRun++; if (whiteRun > maxGap) break; }
            else { cut = y + 1; whiteRun = 0; }
        }
        for (int y = 0; y < cut; ++y) {
            if (colorMask.empty() || colorMask.at<uchar>(H - 1 - y, x) == 0) setWhite(H - 1 - y, x);
        }
    }

    // --- 3. 左边缘 ---
    for (int y = 0; y < H; ++y) {
        int cut = 0;
        int whiteRun = 0;
        for (int x = 0; x < maxScanX; ++x) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, x) > 0) {
                cut = x + 1; whiteRun = 0; continue;
            }
            uchar v = gray.at<uchar>(y, x);
            if (v < darkThr) { cut = x + 1; whiteRun = 0; }
            else if (v > 235) { whiteRun++; if (whiteRun > maxGap) break; }
            else { cut = x + 1; whiteRun = 0; }
        }
        for (int x = 0; x < cut; ++x) {
            if (colorMask.empty() || colorMask.at<uchar>(y, x) == 0) setWhite(y, x);
        }
    }

    // --- 4. 右边缘 ---
    for (int y = 0; y < H; ++y) {
        int cut = 0;
        int whiteRun = 0;
        for (int x = 0; x < maxScanX; ++x) {
            if (!colorMask.empty() && colorMask.at<uchar>(y, W - 1 - x) > 0) {
                cut = x + 1; whiteRun = 0; continue;
            }
            uchar v = gray.at<uchar>(y, W - 1 - x);
            if (v < darkThr) { cut = x + 1; whiteRun = 0; }
            else if (v > 235) { whiteRun++; if (whiteRun > maxGap) break; }
            else { cut = x + 1; whiteRun = 0; }
        }
        for (int x = 0; x < cut; ++x) {
            if (colorMask.empty() || colorMask.at<uchar>(y, W - 1 - x) == 0) setWhite(y, W - 1 - x);
        }
    }

    // ============================================================
    // ★ 依然生成 Debug 图，这是我们的底牌
    // ============================================================
    try {
        cv::Mat debugImg = src.clone();
        if (debugImg.channels() == 1) cv::cvtColor(debugImg, debugImg, cv::COLOR_GRAY2BGR);
        cv::Mat diffMask;
        cv::compare(src, dst, diffMask, cv::CMP_NE);
        cv::Mat diffGray;
        if (diffMask.channels() > 1) cv::cvtColor(diffMask, diffGray, cv::COLOR_BGR2GRAY);
        else diffGray = diffMask;
        
        // 红 = 实际被涂白的像素
        if (debugImg.channels() == 3) debugImg.setTo(cv::Scalar(0, 0, 255), diffGray);
        // 绿 = 印章保护范围
        if (debugImg.channels() == 3) debugImg.setTo(cv::Scalar(0, 255, 0), colorMask);

        QString debugPath = QDir::currentPath() + "/debug_mask_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".jpg";
        cv::imwrite(debugPath.toLocal8Bit().toStdString(), debugImg);
        qDebug() << "【调试信息】已生成调试图:" << debugPath;
    } catch (...) {}
    // ============================================================

    result.image = dst;
    result.filledPixels = paintedCount;
    result.ok = true;
    return result;
}

} // namespace process