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

    // 1. 提取印章保护掩膜
    cv::Mat colorMask;
    if (!options.protectMask.empty()) {
        colorMask = options.protectMask.clone();
    } else if (src.channels() >= 3) {
        // 没传的话，自己找。极度放宽检测条件：红/蓝/绿等高饱和度区域全要
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // 只要饱和度 > 25 的算作印章彩色
        cv::inRange(hsvCh[1], 25, 255, colorMask);
        // 亮度 > 30 的才要，防止把黑色当彩色
        cv::Mat valMask;
        cv::inRange(hsvCh[2], 30, 255, valMask);
        cv::bitwise_and(colorMask, valMask, colorMask);
    }

    // 印章掩膜膨胀，把印章边缘的过渡带也保护起来
    if (!colorMask.empty()) {
        cv::dilate(colorMask, colorMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)));
    }

    // 2. 自适应二值化提取所有暗色区域
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 10);

    // 3. 极致开运算：21x21的核直接把文字、表格线抹除，只留黑边
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::Mat opened;
    cv::morphologyEx(binary, opened, cv::MORPH_OPEN, openKernel);

    // 4. 闭运算：连接黑边的断裂处，防止反光造成断点漏切
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::morphologyEx(opened, opened, cv::MORPH_CLOSE, closeKernel);

    // 5. 寻找轮廓并只保留贴边的部分
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        // 只要触及边缘 40 像素内，或者特别长/特别宽，就认为是黑边
        bool touchesEdge = (r.x <= 40) || (r.y <= 40) || (r.x + r.width >= W - 40) || (r.y + r.height >= H - 40);
        bool isBig = (r.width > 80 || r.height > 80);
        
        // 排除整页大黑图
        bool isWholePage = (r.width >= W - 20 && r.height >= H - 20);

        if (touchesEdge && isBig && !isWholePage) {
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 6. 涂白前稍微膨胀一下掩膜，吃掉黑边外侧发灰的阴影
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));

    // 7. 核心：从涂白掩膜中，绝对扣除印章区域！
    if (!colorMask.empty()) {
        cv::bitwise_and(mask, ~colorMask, mask);
    }

    // 8. 进行纯白填充
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