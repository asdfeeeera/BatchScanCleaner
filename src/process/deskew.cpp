#include "deskew.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

namespace process {

namespace {

// 辅助：按特定角度旋转（用于投影法）
cv::Mat rotateForProjection(const cv::Mat &src, double angle)
{
    const cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, src.size(),
                   cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return dst;
}

// 辅助：投影法搜索角度
std::pair<double, bool> searchAngleByProjection(const cv::Mat &binary,
                                                  double minAngle, double maxAngle, double step)
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

// ★ 借鉴商业软件：找最长直线
std::pair<double, bool> detectAngleByLongestLines(const cv::Mat &gray)
{
    const int W = gray.cols;
    const int H = gray.rows;
    if (W < 100 || H < 100) return { 0.0, false };

    cv::Mat edges;
    // Canny 阈值设高一点，只保留最明显的边缘
    cv::Canny(gray, edges, 50, 150, 3);

    std::vector<cv::Vec4i> lines;
    // ★ 核心修改：要求线段长度至少占页面宽度的 30%
    const int minLineLen = std::max(100, static_cast<int>(std::min(W, H) * 0.3));
    const int maxLineGap = 20;
    
    // 阈值 80：过滤掉噪点短线段
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 80, minLineLen, maxLineGap);

    if (lines.empty()) return { 0.0, false };

    // 收集所有长线的角度
    std::vector<double> angles;
    for (const auto &l : lines) {
        const double dx = static_cast<double>(l[2] - l[0]);
        const double dy = static_cast<double>(l[3] - l[1]);
        if (std::abs(dx) < 1.0 && std::abs(dy) < 1.0) continue;

        // 排除接近垂直的线（只关心水平方向的倾斜）
        if (std::abs(dy) > std::abs(dx) * 1.5) continue;

        double ang = std::atan2(dy, dx) * 180.0 / CV_PI;

        // 归一化到 [-45, 45]
        while (ang >  45.0) ang -= 90.0;
        while (ang < -45.0) ang += 90.0;

        // 过滤掉太陡峭的线（比如超过 30 度的不可能是正常表格倾斜）
        if (std::abs(ang) < 0.1) continue;
        if (std::abs(ang) > 30.0) continue;

        angles.push_back(ang);
    }

    if (angles.size() < 2) return { 0.0, false };

    // 取中位数作为最终角度
    std::sort(angles.begin(), angles.end());
    const double median = angles[angles.size() / 2];

    return { -median, true };
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

    // ★ 第 1 优先级：寻找最长直线（商业软件常用方案）
    auto lineRes = detectAngleByLongestLines(smallGray);
    if (lineRes.second) {
        return lineRes.first;
    }

    // ★ 第 2 优先级：投影法（纯文字）
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
    const double coarseAngle = coarse.first;

    auto fine = searchAngleByProjection(binary, coarseAngle - 1.5, coarseAngle + 1.5, 0.05);
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

    // 小于 1.5 度认为是正的，不转，防止像素损失
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