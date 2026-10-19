#include "deskew.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

namespace process {

namespace {

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

std::pair<double, bool> detectAngleByLongestLines(const cv::Mat &gray)
{
    const int W = gray.cols;
    const int H = gray.rows;
    if (W < 100 || H < 100) return { 0.0, false };

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150, 3);

    std::vector<cv::Vec4i> lines;
    const int minLineLen = std::max(60, static_cast<int>(std::min(W, H) * 0.15));
    const int maxLineGap = 20;
    
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 60, minLineLen, maxLineGap);

    if (lines.empty()) return { 0.0, false };

    std::vector<double> angles;
    for (const auto &l : lines) {
        const double dx = static_cast<double>(l[2] - l[0]);
        const double dy = static_cast<double>(l[3] - l[1]);
        if (std::abs(dx) < 1.0 && std::abs(dy) < 1.0) continue;
        if (std::abs(dy) > std::abs(dx) * 1.5) continue;

        double ang = std::atan2(dy, dx) * 180.0 / CV_PI;
        while (ang >  45.0) ang -= 90.0;
        while (ang < -45.0) ang += 90.0;

        if (std::abs(ang) > 30.0) continue;
        angles.push_back(ang);
    }

    if (angles.size() < 2) return { 0.0, false };

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

    cv::Mat smallGray;
    const int maxDim = std::max(gray.cols, gray.rows);
    if (maxDim > 800) {
        const double scale = 800.0 / maxDim;
        cv::resize(gray, smallGray, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        smallGray = gray;
    }

    auto lineRes = detectAngleByLongestLines(smallGray);
    if (lineRes.second) return lineRes.first;

    cv::Mat binary;
    cv::adaptiveThreshold(smallGray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 10);

    auto coarse = searchAngleByProjection(binary, -30.0, 30.0, 0.5);
    const double coarseAngle = coarse.first;
    auto fine = searchAngleByProjection(binary, coarseAngle - 1.0, coarseAngle + 1.0, 0.05);
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

    const int newW = static_cast<int>(std::ceil(srcH * sinA + srcW * cosA)) + 2;
    const int newH = static_cast<int>(std::ceil(srcH * cosA + srcW * sinA)) + 2;

    cv::Point2f center(srcW / 2.0f, srcH / 2.0f);

    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    rot.at<double>(0, 2) += (newW - srcW) / 2.0;
    rot.at<double>(1, 2) += (newH - srcH) / 2.0;

    cv::Mat dst;
    cv::warpAffine(src, dst, rot, cv::Size(newW, newH), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(255));

    // ★ 旋转后自动裁切四周多余的白边
    cv::Mat gray;
    if (dst.channels() == 3) cv::cvtColor(dst, gray, cv::COLOR_BGR2GRAY);
    else if (dst.channels() == 4) cv::cvtColor(dst, gray, cv::COLOR_BGRA2GRAY);
    else gray = dst;

    cv::Mat mask;
    cv::threshold(gray, mask, 245, 255, cv::THRESH_BINARY_INV);
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    
    if (!points.empty()) {
        cv::Rect rect = cv::boundingRect(points);
        int pad = 5;
        rect.x = std::max(0, rect.x - pad);
        rect.y = std::max(0, rect.y - pad);
        rect.width = std::min(dst.cols - rect.x, rect.width + 2 * pad);
        rect.height = std::min(dst.rows - rect.y, rect.height + 2 * pad);
        
        if (rect.width > 50 && rect.height > 50) {
            dst = dst(rect).clone();
        }
    }

    return dst;
}

DeskewResult Deskew::autoDeskew(const cv::Mat &src)
{
    DeskewResult result;
    if (src.empty()) return result;

    const double angle = detectAngle(src);

    if (std::abs(angle) < 0.3) {
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