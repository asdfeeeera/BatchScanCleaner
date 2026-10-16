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
// ★ 新增：判断这张图是否值得做"白纸泛黄"底色处理
//   是 → 返回 true
//   否（彩色证件/照片/彩色文档）→ 返回 false，跳过处理
// ============================================================
static bool isYellowishPaper(const cv::Mat &bgr)
{
    if (bgr.empty() || bgr.channels() < 3) return true;  // 灰度图默认处理

    // 缩到 200 宽，加速
    cv::Mat small;
    const int maxDim = std::max(bgr.cols, bgr.rows);
    if (maxDim > 200) {
        const double scale = 200.0 / maxDim;
        cv::resize(bgr, small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        small = bgr;
    }

    // 1. 平均 BGR
    cv::Scalar meanBGR = cv::mean(small);
    const double meanB = meanBGR[0];
    const double meanG = meanBGR[1];
    const double meanR = meanBGR[2];

    // 2. 平均饱和度
    cv::Mat hsv;
    cv::cvtColor(small, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar meanHSV = cv::mean(hsv);
    const double meanSat = meanHSV[1];
    const double meanVal = meanHSV[2];

    // 3. 判断
    //   彩色（饱和度 > 25）→ 不处理
    if (meanSat > 25.0) return false;

    //   整体太暗（V < 180）→ 不处理（可能是照片/深色纸）
    if (meanVal < 180.0) return false;

    //   "偏黄"特征：R > G > B 且 R - B >= 3
    //   如果 R <= B 或者差值很小 → 不是泛黄 → 不处理
    const double rbDiff = meanR - meanB;
    if (rbDiff < 3.0) return false;

    //   如果 R - B > 40 → 太黄了（可能是彩色纸）→ 不处理
    if (rbDiff > 40.0) return false;

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

    // ★ 新增：先判断是否是"白纸泛黄"场景
    {
        cv::Mat bgr;
        if (channels == 4) {
            cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        } else if (channels == 3) {
            bgr = src;
        }
        if (!isYellowishPaper(bgr)) {
            // 彩色/照片 → 跳过底色处理
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

    // 纸已经足够白 → 跳过
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