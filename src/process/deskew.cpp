#include "deskew.h"

#include <opencv2/imgproc.hpp>
#include <cmath>

namespace process {

namespace {

// 旋转图像（不扩展画布，用于投影法中间步骤）
cv::Mat rotateForProjection(const cv::Mat &src, double angle)
{
    const cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, src.size(),
                   cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    return dst;
}

} // namespace

double Deskew::detectAngle(const cv::Mat &src)
{
    if (src.empty()) {
        return 0.0;
    }

    // 1. 转灰度
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    // 2. 二值化，文字为白前景
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 3. 降采样，加速迭代（最多 800 像素长边）
    cv::Mat small;
    const int maxDim = std::max(binary.cols, binary.rows);
    if (maxDim > 800) {
        const double scale = 800.0 / maxDim;
        cv::resize(binary, small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        small = binary;
    }

    // 4. 投影法：找让水平投影方差最大的角度
    //    文字行水平时，行与行的空白会让投影方差最大
    double bestAngle = 0.0;
    double bestScore = -1.0;

    for (double angle = -10.0; angle <= 10.0; angle += 0.1) {
        const cv::Mat rotated = rotateForProjection(small, angle);

        // 水平方向投影（每行的前景像素总数）
        cv::Mat projection;
        cv::reduce(rotated, projection, 1, cv::REDUCE_SUM, CV_32S);

        // 计算投影的方差
        cv::Scalar mean, stddev;
        cv::meanStdDev(projection, mean, stddev);
        const double score = stddev[0] * stddev[0];

        if (score > bestScore) {
            bestScore = score;
            bestAngle = angle;
        }
    }

    return bestAngle;
}

cv::Mat Deskew::rotateKeepAll(const cv::Mat &src, double angle)
{
    if (std::abs(angle) < 0.001) {
        return src.clone();
    }

    const int srcW = src.cols;
    const int srcH = src.rows;

    // 计算旋转后需要的画布大小，确保内容完整
    const double rad = angle * CV_PI / 180.0;
    const double cosA = std::abs(std::cos(rad));
    const double sinA = std::abs(std::sin(rad));

    const int newW = static_cast<int>(srcH * sinA + srcW * cosA);
    const int newH = static_cast<int>(srcH * cosA + srcW * sinA);

    // 旋转中心：原图中心
    cv::Point2f center(srcW / 2.0f, srcH / 2.0f);

    // 让旋转后的内容居中
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    rot.at<double>(0, 2) += (newW - srcW) / 2.0;
    rot.at<double>(1, 2) += (newH - srcH) / 2.0;

    cv::Mat dst;
    cv::warpAffine(src, dst, rot, cv::Size(newW, newH),
                   cv::INTER_CUBIC,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(255, 255, 255));
    return dst;
}

DeskewResult Deskew::autoDeskew(const cv::Mat &src)
{
    DeskewResult result;

    if (src.empty()) {
        return result;
    }

    const double angle = detectAngle(src);

    if (std::abs(angle) < 0.2) {
        result.image = src.clone();
        result.angle = 0.0;
        result.ok = true;
        return result;
    }

    if (std::abs(angle) > 10.0) {
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