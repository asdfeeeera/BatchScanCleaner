#include "deskew.h"

#include <opencv2/imgproc.hpp>
#include <cmath>

namespace process {

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

    // 3. 轻微膨胀，让文字连成块，便于角度检测
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(binary, binary, kernel);

    // 4. 找所有前景像素点
    std::vector<cv::Point> points;
    cv::findNonZero(binary, points);

    if (points.size() < 100) {
        // 前景太少，无法判断
        return 0.0;
    }

    // 5. 计算最小外接矩形，得到角度
    cv::RotatedRect rect = cv::minAreaRect(points);
    double angle = rect.angle;

    // minAreaRect 返回的角度范围是 [-90, 0)
    // 转换为 [-45, 45] 区间，表示相对水平的倾斜角
    if (angle < -45.0) {
        angle += 90.0;
    } else if (angle > 45.0) {
        angle -= 90.0;
    }

    return angle;
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

    // 倾斜角太小，跳过
    if (std::abs(angle) < 0.3) {
        result.image = src.clone();
        result.angle = 0.0;
        result.ok = true;
        return result;
    }

    // 角度过大（超过 30 度），可能识别错误，跳过
    if (std::abs(angle) > 30.0) {
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