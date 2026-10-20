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

    // 1. 提取极高饱和度的彩色掩膜（不仅抓红色，蓝色签名等也抓），做超大膨胀
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // ★ 饱和度 > 20 就算彩色，极其宽松！
        cv::inRange(hsvCh[1], 20, 255, colorMask);
        
        // 亮度 > 20，防止把黑色当彩色
        cv::Mat valMask;
        cv::inRange(hsvCh[2], 20, 255, valMask);
        cv::bitwise_and(colorMask, valMask, colorMask);
        
        // ★ 超级膨胀，把印章周围 25 个像素全部保护起来，绝不切到印章一丝一毫！
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    // 2. 自适应二值化找暗色区域（阈值定在 150，把灰影也抓出来）
    cv::Mat binary;
    cv::threshold(gray, binary, 150, 255, cv::THRESH_BINARY_INV);

    // 3. 超强闭运算（30x30），把黑边里的反光、断点全部连成一大整块
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, closeKernel);

    // 4. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        
        // 判定：必须在边缘 30 像素内，并且面积要够大
        bool touchesEdge = (r.x <= 30) || (r.y <= 30) || 
                           (r.x + r.width >= W - 30) || (r.y + r.height >= H - 30);
        
        bool isBigEnough = (r.width > 40 || r.height > 40);

        // 排除整页是黑图的情况
        bool isWholePage = (r.width >= W - 10 && r.height >= H - 10);

        if (touchesEdge && isBigEnough && !isWholePage) {
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 5. 膨胀掩膜，确保边缘灰色过渡带全被涂白
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
    cv::dilate(mask, mask, dilateKernel);

    // 6. 核心：强制扣除彩色保护区域
    if (!colorMask.empty()) {
        cv::bitwise_and(mask, ~colorMask, mask);
    }

    // 7. 纯涂白（绝不改变画布）
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