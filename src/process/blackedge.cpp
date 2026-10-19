#include "blackedge.h"

#include <opencv2/imgproc.hpp>
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

    // 1. 提取彩色印章保护掩膜
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        cv::inRange(hsvCh[1], 40, 255, colorMask);
        cv::dilate(colorMask, colorMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    }

    // 2. 二值化（大津法）
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 3. 超强开运算，把文字和表格线彻底抹除，只留下大块的黑边和照片
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::Mat opened;
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, openKernel);

    // 4. 闭运算，连接黑边内部的断点
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(opened, opened, cv::MORPH_CLOSE, closeKernel);

    // 5. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        
        // 判定：必须在边缘 20 像素内，并且块不能太小
        bool touchesEdge = (r.x <= 20) || (r.y <= 20) || 
                           (r.x + r.width >= W - 20) || (r.y + r.height >= H - 20);
        
        // 面积过滤，避免留下微小的边缘噪点
        bool isBigEnough = (r.width > 30 || r.height > 30);

        // 排除整页是黑图的情况（比如原图本身就有大片黑色背景）
        bool isWholePage = (r.width >= W - 10 && r.height >= H - 10);

        if (touchesEdge && isBigEnough && !isWholePage) {
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 6. 膨胀掩膜，确保边缘的灰色过渡带也能被涂干净
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(mask, mask, dilateKernel);

    // 7. 扣除印章
    if (!colorMask.empty()) {
        cv::bitwise_and(mask, ~colorMask, mask);
    }

    // 8. 涂白（绝不用 Rect 裁剪，画布永不缩小）
    cv::Mat dst = src.clone();
    cv::Scalar white;
    if (src.channels() == 4) white = cv::Scalar(255, 255, 255, 255);
    else if (src.channels() == 3) white = cv::Scalar(255, 255, 255);
    else white = cv::Scalar(255);

    dst.setTo(white, mask);

    result.image = dst;
    result.filledPixels = cv::countNonZero(mask);
    result.ok = true;
    return result;
}

} // namespace process