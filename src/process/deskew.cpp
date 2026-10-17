#include "deskew.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

namespace process {

namespace {

// ★★★ 最强方案：边缘轮廓四边形检测（专治各种倾斜纸张、证件、拍照） ★★★
std::pair<double, bool> detectAngleByQuad(const cv::Mat &gray)
{
    cv::Mat blurred, edges;
    // 1. 模糊去噪
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    
    // 2. 边缘检测（Canny）
    cv::Canny(blurred, edges, 50, 150);

    // 3. 膨胀边缘，把断开的纸张边缘连成一个闭环
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(edges, edges, kernel);

    // 4. 寻找所有轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    double imgArea = gray.cols * gray.rows;
    
    // 5. 按面积从大到小排序
    std::sort(contours.begin(), contours.end(), 
              [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                  return cv::contourArea(c1) > cv::contourArea(c2);
              });

    // 6. 遍历最大的几个轮廓，找四边形
    for (size_t i = 0; i < std::min(contours.size(), (size_t)5); ++i) {
        double area = cv::contourArea(contours[i]);
        
        // 纸张面积必须足够大（大于全图 15%）
        if (area < imgArea * 0.15) continue;

        std::vector<cv::Point> approx;
        double peri = cv::arcLength(contours[i], true);
        // 多边形逼近
        cv::approxPolyDP(contours[i], approx, 0.02 * peri, true);

        // 如果逼近结果正好是 4 个点（四边形）
        if (approx.size() == 4) {
            cv::RotatedRect minRect = cv::minAreaRect(approx);
            double angle = minRect.angle;
            
            // 归一化到 [-45, 45]
            if (angle > 45.0) angle -= 90.0;
            if (angle < -45.0) angle += 90.0;

            // 宽高必须合理
            if (minRect.size.width < 50 || minRect.size.height < 50) continue;

            // 只要倾角大于 0.5 度，立刻返回
            if (std::abs(angle) > 0.5) {
                return { -angle, true }; // 符号反转用于旋转
            } else {
                return { 0.0, true }; // 说明是正的
            }
        }
    }
    return { 0.0, false };
}

// 兜底：投影法（用于纯文字、没有明显边界的图）
cv::Mat rotateForProjection(const cv::Mat &src, double angle)
{
    const cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, src.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return dst;
}

std::pair<double, bool> searchAngleByProjection(const cv::Mat &binary, double minAngle, double maxAngle, double step)
{
    double bestAngle = 0.0;
    double bestScore = -1.0;
    double secondScore = -1.0;

    for (double angle = minAngle; angle <= maxAngle + 1e-6; angle += step) {
        const cv::Mat rotated = rotateForProjection(binary, angle);
        cv::Mat projection;
        cv::reduce(rotated, projection, 1, cv::REDUCE_SUM, CV_32S);
        cv::Scalar mean, stddev;
        cv::meanStdDev(projection, mean, stddev);
        const double score = stddev[0] * stddev[0];

        if (score > bestScore) {
            secondScore = bestScore;
            bestScore = score;
            bestAngle = angle;
        } else if (score > secondScore) {
            secondScore = score;
        }
    }

    bool confident = false;
    if (bestScore > 0.0) {
        if (secondScore > 0.0) confident = (bestScore > secondScore * 1.15);
        else confident = true;
        if (bestScore < 500.0) confident = false;
    }
    return { bestAngle, confident };
}

} // namespace

double Deskew::detectAngle(const cv::Mat &src)
{
    if (src.empty()) return 0.0;

    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src.clone();

    // 缩放加速处理
    cv::Mat smallGray;
    const int maxDim = std::max(gray.cols, gray.rows);
    if (maxDim > 800) {
        const double scale = 800.0 / maxDim;
        cv::resize(gray, smallGray, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        smallGray = gray;
    }

    // ★ 第一优先级：找纸张/证件的四边形边缘
    auto quadRes = detectAngleByQuad(smallGray);
    if (quadRes.second) {
        return quadRes.first;
    }

    // ★ 第二优先级：投影法（纯文字页）
    const int W = smallGray.cols;
    const int H = smallGray.rows;
    const int mx = static_cast<int>(W * 0.15);
    const int my = static_cast<int>(H * 0.15);
    cv::Rect centerRoi(mx, my, W - 2 * mx, H - 2 * my);
    centerRoi &= cv::Rect(0, 0, W, H);

    cv::Mat center = smallGray(centerRoi).clone();
    cv::Mat binary;
    cv::threshold(center, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    auto coarse = searchAngleByProjection(binary, -45.0, 45.0, 1.0);
    auto fine = searchAngleByProjection(binary, coarse.first - 1.5, coarse.first + 1.5, 0.05);
    auto result = fine.second ? fine : coarse;

    return result.first;
}

cv::Mat Deskew::rotateKeepAll(const cv::Mat &src, double angle)
{
    if (std::abs(angle) < 0.001) return src.clone();

    const int srcW = src.cols;
    const int srcH = src.rows;
    const double rad = angle * CV_PI / 180.0;
    const double cosA = std::abs(std::cos(rad));
    const double sinA = std::abs(std::sin(rad));
    const int newW = static_cast<int>(srcH * sinA + srcW * cosA);
    const int newH = static_cast<int>(srcH * cosA + srcW * sinA);

    cv::Point2f center(srcW / 2.0f, srcH / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    rot.at<double>(0, 2) += (newW - srcW) / 2.0;
    rot.at<double>(1, 2) += (newH - srcH) / 2.0;

    cv::Mat dst;
    cv::warpAffine(src, dst, rot, cv::Size(newW, newH),
                   cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));
    return dst;
}

DeskewResult Deskew::autoDeskew(const cv::Mat &src)
{
    DeskewResult result;
    if (src.empty()) return result;

    const double angle = detectAngle(src);

    if (std::abs(angle) < 1.5) {
        result.image = src.clone();
        result.angle = 0.0;
        result.ok = true;
        return result;
    }

    if (std::abs(angle) > 45.0) {
        result.image = src.clone();
        result.angle = 0.0;
        result.ok = false;
        return result;
    }

    result.image = rotateKeepAll(src, angle);
    result.angle = angle;
    result.ok = true;
    return result;
}

} // namespace process