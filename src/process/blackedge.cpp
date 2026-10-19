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

    // 1. 提取印章绝对保护区域
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
        
        // 稍微膨胀，绝对防止切到印章边缘
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::dilate(colorMask, colorMask, dilateKernel);
    }

    const double paperGray = calcPaperGray(gray);
    result.paperGray = paperGray;

    // 2. 二值化，把偏灰的黑边、黑图、文字全部抓出来
    cv::Mat binary;
    // 阈值 130 足够抓住发灰的阴影，又不会把白纸抓进去
    cv::threshold(gray, binary, 130, 255, cv::THRESH_BINARY_INV);

    // 3. 形态学闭运算：把黑边里因为反光产生的白点、断点全部连接起来，变成实心大块
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, closeKernel);

    // 4. 给图像加一圈虚拟黑边，确保贴边的黑边能稳稳地和图像边缘连通
    int pad = 5;
    cv::Mat paddedBin;
    cv::copyMakeBorder(binary, paddedBin, pad, pad, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(0));

    // 5. 寻找所有的连通域
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(paddedBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    cv::Mat paddedMask;
    cv::copyMakeBorder(mask, paddedMask, pad, pad, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(0));

    for (const auto &cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        
        // 还原到原图的坐标系判定是否贴边（容忍 15 像素的轻微白边）
        bool touchLeft   = (r.x <= pad + 15);
        bool touchRight  = (r.x + r.width >= W + pad - 15);
        bool touchTop    = (r.y <= pad + 15);
        bool touchBottom = (r.y + r.height >= H + pad - 15);
        
        bool touchesEdge = touchLeft || touchRight || touchTop || touchBottom;
        
        // 判定尺寸：如果它足够长（超过图像宽或高的 10%），大概率是黑边
        bool isLongEnough = (r.width > W * 0.10) || (r.height > H * 0.10);

        // 判定实心度：黑边是实心的，文字是镂空的
        double contourArea = cv::contourArea(cnt);
        double rectArea = r.width * r.height;
        double fillRatio = (rectArea > 0) ? (contourArea / rectArea) : 0;

        // 判定：只要贴边，且（足够长 或 非常实心），就认定为黑边！
        if (touchesEdge && (isLongEnough || fillRatio > 0.8)) {
            
            // 排除整页是黑图的情况（如果这个框几乎等于全图，不要处理）
            if (r.width >= W - 10 && r.height >= H - 10) continue;

            // 认定是黑边，把这个轮廓画到掩膜上
            cv::drawContours(paddedMask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 裁掉 padding
    cv::Rect roi(pad, pad, W, H);
    cv::Mat finalMask = paddedMask(roi).clone();

    // 6. 对掩膜进行膨胀，吃掉黑边与纸张之间的过渡灰带（保证切得干干净净）
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(finalMask, finalMask, dilateKernel);

    // 7. 扣除印章区域
    if (!colorMask.empty()) {
        cv::bitwise_and(finalMask, ~colorMask, finalMask);
    }

    // 8. 将最终的黑边区域涂成纯白
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