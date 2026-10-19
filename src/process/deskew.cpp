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
        if (secondScore > 0.0) confident = (bestScore > secondScore * 1.2);
        else confident = true;
        if (bestScore < 1000.0) confident = false;
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

    cv::Mat smallGray;
    const int maxDim = std::max(gray.cols, gray.rows);
    if (maxDim > 1000) {
        const double scale = 1000.0 / maxDim;
        cv::resize(gray, smallGray, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        smallGray = gray;
    }

    cv::Mat binary;
    cv::adaptiveThreshold(smallGray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 31, 15);
    cv::GaussianBlur(binary, binary, cv::Size(3, 3), 0);

    auto coarse = searchAngleByProjection(binary, -10.0, 10.0, 0.2);
    const double coarseAngle = coarse.first;

    auto fine = searchAngleByProjection(binary, coarseAngle - 0.5, coarseAngle + 0.5, 0.05);
    
    if (coarse.second) return coarseAngle;
    return fine.second ? fine.first : coarseAngle;
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

    // ★ 绝对没有裁剪代码，画布尺寸一毫米都不会缩水
    return dst;
}

DeskewResult Deskew::autoDeskew(const cv::Mat &src)
{
    DeskewResult result;
    if (src.empty()) return result;

    const double angle = detectAngle(src);

    if (std::abs(angle) < 0.2) {
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