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

    // 1. 提取极高饱和度的红色印章掩膜，并做极大的膨胀，确保印章附近绝不涂白
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // 提取红色区域（S > 40）
        cv::inRange(hsvCh[1], 40, 255, colorMask);
        
        // ★ 超级膨胀，把印章周围 20 个像素都保护起来！
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    // 2. 二值化找暗色区域（大津法自适应，不管黑边还是发灰的阴影都能抓出来）
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 3. 超强开运算，把正文的文字、表格线彻底抹除，只留下大块的黑边和照片
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::Mat opened;
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, openKernel);

    // 4. 闭运算，连接黑边内部的断点
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::morphologyEx(opened, opened, cv::MORPH_CLOSE, closeKernel);

    // 5. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        
        // 判定：必须在边缘 30 像素内，并且块不能太小
        bool touchesEdge = (r.x <= 30) || (r.y <= 30) || 
                           (r.x + r.width >= W - 30) || (r.y + r.height >= H - 30);
        
        // 面积过滤，避免留下微小的边缘噪点
        bool isBigEnough = (r.width > 50 || r.height > 50);

        // 排除整页是黑图的情况
        bool isWholePage = (r.width >= W - 10 && r.height >= H - 10);

        if (touchesEdge && isBigEnough && !isWholePage) {
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 6. 膨胀掩膜，确保边缘的灰色过渡带也能被涂干净
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
    cv::dilate(mask, mask, dilateKernel);

    // 7. 扣除印章保护区域
    if (!colorMask.empty()) {
        cv::bitwise_and(mask, ~colorMask, mask);
    }

    // 8. ★ 绝对不裁剪！纯涂白！
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