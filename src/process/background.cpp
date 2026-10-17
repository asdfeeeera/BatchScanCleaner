#include "background.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

namespace process {

// ============================================================
// 估算纸底灰度
// ============================================================
double Background::estimatePaperGray(const cv::Mat &gray,
                                      const BackgroundOptions &options)
{
    const int W = gray.cols;
    const int H = gray.rows;
    const int mx = static_cast<int>(W * (1.0 - options.paperSampleRatio) / 2.0);
    const int my = static_cast<int>(H * (1.0 - options.paperSampleRatio) / 2.0);

    std::vector<uchar> pixels;
    pixels.reserve(static_cast<size_t>(W) * H / 8);

    for (int y = my; y < H - my; y += 2) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = mx; x < W - mx; x += 2) {
            pixels.push_back(row[x]);
        }
    }

    if (pixels.empty()) {
        cv::Scalar m = cv::mean(gray);
        return m[0];
    }

    const size_t idx = static_cast<size_t>(pixels.size() * options.paperPercentile);
    const size_t safeIdx = std::min(idx, pixels.size() - 1);
    std::nth_element(pixels.begin(), pixels.begin() + safeIdx, pixels.end());
    return static_cast<double>(pixels[safeIdx]);
}

// ============================================================
// 判断是否是纯白泛黄纸（严格版）
// ============================================================
static bool isYellowishPaper(const cv::Mat &bgr)
{
    if (bgr.empty() || bgr.channels() < 3) return true;

    cv::Mat small;
    const int maxDim = std::max(bgr.cols, bgr.rows);
    if (maxDim > 200) {
        const double scale = 200.0 / maxDim;
        cv::resize(bgr, small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        small = bgr;
    }

    cv::Scalar meanBGR = cv::mean(small);
    const double meanB = meanBGR[0];
    const double meanG = meanBGR[1];
    const double meanR = meanBGR[2];

    cv::Mat hsv;
    cv::cvtColor(small, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar meanHSV = cv::mean(hsv);
    const double meanSat = meanHSV[1];
    const double meanVal = meanHSV[2];

    // 条件 1：饱和度必须极低（粉色证件饱和度会偏高）
    if (meanSat > 15.0) return false;

    // 条件 2：整体必须足够亮
    if (meanVal < 200.0) return false;

    // 条件 3："偏黄"特征
    const double rbDiff = meanR - meanB;
    if (rbDiff < 2.0) return false;  // 偏蓝或纯白
    if (rbDiff > 35.0) return false; // 太黄了

    return true;
}

// ============================================================
// 主处理
// ============================================================
BackgroundResult Background::whiten(const cv::Mat &src,
                                     const BackgroundOptions &options)
{
    BackgroundResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;
    const int channels = src.channels();

    // ★ 超级拦截：只要有 3% 的像素是明显彩色（红章、蓝底、粉底），直接跳过
    if (channels >= 3) {
        cv::Mat bgr;
        if (channels == 4) cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        else bgr = src;

        cv::Mat hsv, satMask;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        
        // 饱和度 > 40 视为明显的彩色
        cv::inRange(hsvCh[1], 40, 255, satMask);
        double coloredRatio = static_cast<double>(cv::countNonZero(satMask)) / (W * H);
        
        if (coloredRatio > 0.03) { // 3% 阈值
            result.image = src.clone();
            result.skipped = true;
            result.ok = true;
            return result;
        }
    }

    // 黑白/泛黄判断
    {
        cv::Mat bgr;
        if (channels == 4) {
            cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        } else if (channels == 3) {
            bgr = src;
        }
        if (!isYellowishPaper(bgr)) {
            result.image = src.clone();
            result.skipped = true;
            result.ok = true;
            return result;
        }
    }

    // 灰度转换
    cv::Mat gray;
    if (channels == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (channels == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = src.clone();
    }

    const double paperGray = estimatePaperGray(gray, options);
    result.paperGray = paperGray;

    if (paperGray >= 254.5) {
        result.image = src.clone();
        result.skipped = true;
        result.ok = true;
        return result;
    }

    const double contentThreshold = paperGray * options.contentRatio;
    const double safeThreshold = paperGray * 0.55;
    const double effectiveThreshold = std::min(contentThreshold, safeThreshold);

    cv::Mat contentMask;
    cv::threshold(gray, contentMask, effectiveThreshold, 255, cv::THRESH_BINARY_INV);

    // 形态学保护表格线
    {
        cv::Mat darkBin;
        const double darkThr = paperGray * 0.9;
        cv::threshold(gray, darkBin, darkThr, 255, cv::THRESH_BINARY_INV);

        cv::Mat closeKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat connected;
        cv::morphologyEx(darkBin, connected, cv::MORPH_CLOSE, closeKernel);

        cv::Mat hKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(40, 1));
        cv::Mat hLines;
        cv::morphologyEx(connected, hLines, cv::MORPH_OPEN, hKernel);

        cv::Mat vKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(1, 40));
        cv::Mat vLines;
        cv::morphologyEx(connected, vLines, cv::MORPH_OPEN, vKernel);

        cv::Mat linesMask;
        cv::bitwise_or(hLines, vLines, linesMask);

        cv::Mat dilateKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(linesMask, linesMask, dilateKernel);

        cv::bitwise_or(contentMask, linesMask, contentMask);
    }

    // 保护彩色内容（红章、蓝签）
    if (options.protectColor && channels >= 3) {
        cv::Mat bgr;
        if (channels == 4) {
            cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        } else {
            bgr = src;
        }

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> hsvCh;
        cv::split(hsv, hsvCh);
        const cv::Mat &S = hsvCh[1];
        const cv::Mat &V = hsvCh[2];

        cv::Mat colorMask;
        cv::inRange(S, options.colorSatMin, 255, colorMask);

        cv::Mat valMask;
        cv::inRange(V, 40, 255, valMask);
        cv::bitwise_and(colorMask, valMask, colorMask);

        cv::bitwise_or(contentMask, colorMask, contentMask);
    }

    // 外部保护掩膜
    if (!options.protectMask.empty()) {
        cv::Mat ext = options.protectMask;
        if (ext.size() != gray.size()) {
            cv::Mat tmp;
            cv::resize(ext, tmp, gray.size(), 0, 0, cv::INTER_NEAREST);
            ext = tmp;
        }
        if (ext.type() != CV_8UC1) {
            cv::Mat tmp;
            ext.convertTo(tmp, CV_8UC1);
            ext = tmp;
        }
        cv::bitwise_or(contentMask, ext, contentMask);
    }

    cv::Mat bgMask;
    cv::bitwise_not(contentMask, bgMask);

    cv::Mat dst = src.clone();
    cv::Scalar white;
    const int tg = options.targetPaperGray;

    if (channels == 4) {
        white = cv::Scalar(tg, tg, tg, 255);
    } else if (channels == 3) {
        white = cv::Scalar(tg, tg, tg);
    } else {
        white = cv::Scalar(tg);
    }

    dst.setTo(white, bgMask);

    result.image = dst;
    result.whitenedPixels = cv::countNonZero(bgMask);
    result.ok = true;
    return result;
}

} // namespace process