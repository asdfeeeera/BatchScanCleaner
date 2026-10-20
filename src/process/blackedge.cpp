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
        
        // ★ 极度宽松的颜色检测：只要饱和度 > 15（有一丁点颜色）就抓出来
        cv::Mat colorMask2;
        cv::inRange(hsvCh[1], 15, 255, colorMask2);
        cv::Mat valMask;
        cv::inRange(hsvCh[2], 20, 255, valMask); // 亮度 > 20，防止把黑色的字当成彩色
        cv::bitwise_and(colorMask2, valMask, colorMask2);
        
        if (colorMask.empty()) colorMask = colorMask2;
        else cv::bitwise_or(colorMask, colorMask2, colorMask);
    }

    // 2. ★ 物理级护盾：向外疯狂膨胀 35 像素！确保印章周围如同铁板一块
    if (!colorMask.empty()) {
        cv::dilate(colorMask, colorMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(35, 35)));
    }

    // 3. 自适应二值化抓暗区
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 10);

    // 4. 形态学去除文字，连接黑边断点
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::Mat opened;
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, openKernel);

    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::morphologyEx(opened, opened, cv::MORPH_CLOSE, closeKernel);

    // 5. 找轮廓并筛选贴边的
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        bool touchesEdge = (r.x <= 40) || (r.y <= 40) || (r.x + r.width >= W - 40) || (r.y + r.height >= H - 40);
        bool isBig = (r.width > 80 || r.height > 80);
        bool isWholePage = (r.width >= W - 20 && r.height >= H - 20);

        if (touchesEdge && isBig && !isWholePage) {
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 6. 黑边掩膜稍微膨胀，吃掉边缘灰影
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));

    // 7. 核心：从黑边中，强行抠掉那个膨胀了 35 像素的印章护盾！
    if (!colorMask.empty()) {
        cv::bitwise_and(mask, ~colorMask, mask);
    }

    // 8. 纯涂白
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