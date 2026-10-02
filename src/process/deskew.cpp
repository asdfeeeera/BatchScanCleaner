#include "deskew.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

namespace process {

namespace {

cv::Mat rotateForProjection(const cv::Mat &src, double angle)
{
    const cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, src.size(),
                   cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT,
                   cv::Scalar::all(0));
    return dst;
}

std::pair<double, bool> searchAngleByProjection(const cv::Mat &binary,
                                                  double minAngle,
                                                  double maxAngle,
                                                  double step)
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
        if (secondScore > 0.0) {
            confident = (bestScore > secondScore * 1.15);
        } else {
            confident = true;
        }
        if (bestScore < 500.0) {
            confident = false;
        }
    }

    return { bestAngle, confident };
}

cv::Mat roughCropEdges(const cv::Mat &binary)
{
    const int W = binary.cols;
    const int H = binary.rows;

    auto isBrightRow = [&](int y) {
        const uchar *row = binary.ptr<uchar>(y);
        int whiteCount = 0;
        for (int x = 0; x < W; ++x) {
            if (row[x] > 0) ++whiteCount;
        }
        return whiteCount > W * 0.05;
    };

    auto isBrightCol = [&](int x) {
        int whiteCount = 0;
        for (int y = 0; y < H; ++y) {
            if (binary.at<uchar>(y, x) > 0) ++whiteCount;
        }
        return whiteCount > H * 0.05;
    };

    const int maxScanY = static_cast<int>(H * 0.35);
    const int maxScanX = static_cast<int>(W * 0.35);

    int top = 0;
    for (int y = 0; y < maxScanY; ++y) {
        if (isBrightRow(y)) { top = y; break; }
    }
    int bottom = 0;
    for (int i = 0; i < maxScanY; ++i) {
        const int y = H - 1 - i;
        if (isBrightRow(y)) { bottom = i; break; }
    }
    int left = 0;
    for (int x = 0; x < maxScanX; ++x) {
        if (isBrightCol(x)) { left = x; break; }
    }
    int right = 0;
    for (int i = 0; i < maxScanX; ++i) {
        const int x = W - 1 - i;
        if (isBrightCol(x)) { right = i; break; }
    }

    int x0 = left;
    int y0 = top;
    int x1 = W - right;
    int y1 = H - bottom;

    if (x1 - x0 < W * 0.2 || y1 - y0 < H * 0.2) {
        return binary.clone();
    }

    cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
    roi &= cv::Rect(0, 0, W, H);
    return binary(roi).clone();
}

} // namespace

double Deskew::detectAngle(const cv::Mat &src)
{
    if (src.empty()) {
        return 0.0;
    }

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    const int W = gray.cols;
    const int H = gray.rows;
    const int mx = static_cast<int>(W * 0.15);
    const int my = static_cast<int>(H * 0.15);
    cv::Rect centerRoi(mx, my, W - 2 * mx, H - 2 * my);
    centerRoi &= cv::Rect(0, 0, W, H);

    cv::Mat center = gray(centerRoi).clone();

    cv::Mat binary;
    cv::threshold(center, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat small;
    const int maxDim = std::max(binary.cols, binary.rows);
    if (maxDim > 800) {
        const double scale = 800.0 / maxDim;
        cv::resize(binary, small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        small = binary;
    }

    // 粗到细搜索
    auto coarse = searchAngleByProjection(small, -45.0, 45.0, 1.0);
    const double coarseAngle = coarse.first;

    auto fine = searchAngleByProjection(small,
                                        coarseAngle - 1.5,
                                        coarseAngle + 1.5,
                                        0.05);

    auto result = fine.second ? fine : coarse;

    if (!result.second) {
        cv::Mat cropped = roughCropEdges(small);
        if (cropped.cols != small.cols || cropped.rows != small.rows) {
            auto coarse2 = searchAngleByProjection(cropped, -45.0, 45.0, 1.0);
            auto fine2 = searchAngleByProjection(cropped,
                                                  coarse2.first - 1.5,
                                                  coarse2.first + 1.5,
                                                  0.05);
            auto retry = fine2.second ? fine2 : coarse2;
            if (retry.second) {
                return retry.first;
            }
        }
    }

    return result.first;
}

cv::Mat Deskew::rotateKeepAll(const cv::Mat &src, double angle)
{
    if (std::abs(angle) < 0.001) {
        return src.clone();
    }

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
                   cv::INTER_CUBIC,
                   cv::BORDER_CONSTANT,
                   cv::Scalar::all(255));
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

    // ★ 这里从 10.0 改成 45.0
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