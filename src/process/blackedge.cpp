#include "blackedge.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

namespace {

double calcPaperGray(const cv::Mat &gray)
{
    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(gray.cols) * gray.rows / 4);

    for (int y = 0; y < gray.rows; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; x += 2) {
            pixels.push_back(row[x]);
        }
    }

    if (pixels.empty()) return 255.0;

    const size_t idx = static_cast<size_t>(pixels.size() * 0.85);
    std::nth_element(pixels.begin(), pixels.begin() + idx, pixels.end());
    return static_cast<double>(pixels[idx]);
}

} // namespace

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

    // 1. 提取红色印章，绝对保护
    cv::Mat colorMask;
    if (src.channels() >= 3) {
        cv::Mat bgr;
        if (src.channels() == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;
        
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        cv::inRange(hsvCh[1], 30, 255, colorMask);
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    const double paperGray = calcPaperGray(gray);
    result.paperGray = paperGray;

    // 黑度阈值，只要比纸张暗很多就算
    double darkThr = std::min(paperGray * options.darkRatio, 130.0);
    if (darkThr < 40.0) darkThr = 40.0;
    result.darkThreshold = darkThr;

    // 2. 二值化：把偏黑的地方全部抓出来
    cv::Mat binary;
    cv::threshold(gray, binary, darkThr, 255, cv::THRESH_BINARY_INV);

    // 3. 开运算：去掉细小的文字干扰，只留大块黑边
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, openKernel);

    // 4. 闭运算：把黑边里的反光、白点连起来，变成一整块
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, closeKernel);

    // 5. 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);

    for (const auto &cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        
        // 判定是否贴边（允许 15 像素误差）
        bool touchesEdge = (r.x <= 15) || (r.y <= 15) || 
                           (r.x + r.width >= W - 15) || (r.y + r.height >= H - 15);
        
        // 判定面积是否足够大（至少 0.1% 的图幅，过滤掉噪点）
        double area = cv::contourArea(cnt);
        bool isBigEnough = (area > W * H * 0.001);

        // 排除全图（如果是大面积黑图）
        bool isWholePage = (r.width >= W - 10 && r.height >= H - 10);

        if (touchesEdge && isBigEnough && !isWholePage) {
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 6. 膨胀掩膜（保证切干净发灰的过渡带）
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(mask, mask, dilateKernel);

    // 7. 扣除印章区域
    if (!colorMask.empty()) {
        cv::bitwise_and(mask, ~colorMask, mask);
    }

    // 8. 将最终的黑边区域涂成纯白（绝不缩小画布）
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