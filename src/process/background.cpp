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
// 主处理：基于类型的智能底色净化
// ============================================================
BackgroundResult Background::whiten(const cv::Mat &src,
                                     const BackgroundOptions &options)
{
    BackgroundResult result;
    if (src.empty()) return result;

    const int W = src.cols;
    const int H = src.rows;
    const int channels = src.channels();

    // 1. 统一转换为 BGR 用于特征提取，保留 src 原图
    cv::Mat bgr;
    if (channels == 4) {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    } else if (channels == 3) {
        bgr = src;
    } else {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    }

    // 2. 转换为 HSV 提取颜色特征
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvCh;
    cv::split(hsv, hsvCh);
    const cv::Mat &S = hsvCh[1];
    const cv::Mat &V = hsvCh[2];

    // 3. 判定是否为彩色证件（如粉色身份证、护照）
    // 统计高饱和度（S > 50）的像素占比
    cv::Mat coloredMask;
    cv::inRange(S, 50, 255, coloredMask);
    double colorRatio = static_cast<double>(cv::countNonZero(coloredMask)) / (W * H);

    // 如果彩色面积占比超过 12%，判定为彩色证件，直接跳过不处理，保护证件底色
    if (colorRatio > 0.12) {
        result.image = src.clone();
        result.skipped = true;
        result.ok = true;
        return result;
    }

    // 4. 判定是否为泛黄纸张（仅分析低饱和度区域，排除印章干扰）
    cv::Mat lowSatMask;
    cv::inRange(S, 0, 45, lowSatMask);
    
    if (cv::countNonZero(lowSatMask) > 0) {
        cv::Scalar meanBGR = cv::mean(bgr, lowSatMask);
        cv::Scalar meanHSV = cv::mean(hsv, lowSatMask);
        
        double meanB = meanBGR[0];
        double meanR = meanBGR[2];
        double meanVal = meanHSV[2];
        double rbDiff = meanR - meanB;

        // 如果不偏黄（R-B <= 2）或者已经足够白（Val >= 248），则跳过白化
        if (rbDiff <= 2.0 || meanVal >= 248.0) {
            result.image = src.clone();
            result.skipped = true;
            result.ok = true;
            return result;
        }
    } else {
        // 找不到低饱和度区域（全彩图），直接跳过
        result.image = src.clone();
        result.skipped = true;
        result.ok = true;
        return result;
    }

    // 5. 灰度转换，估算纸底灰度
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    const double paperGray = estimatePaperGray(gray, options);
    result.paperGray = paperGray;

    if (paperGray >= 254.5) {
        result.image = src.clone();
        result.skipped = true;
        result.ok = true;
        return result;
    }

    // 6. 生成内容掩膜（文字、表格线、印章）
    const double contentThreshold = paperGray * options.contentRatio;
    const double safeThreshold = paperGray * 0.55;
    const double effectiveThreshold = std::min(contentThreshold, safeThreshold);

    cv::Mat contentMask;
    cv::threshold(gray, contentMask, effectiveThreshold, 255, cv::THRESH_BINARY_INV);

    // 保护表格线
    {
        cv::Mat darkBin;
        const double darkThr = paperGray * 0.9;
        cv::threshold(gray, darkBin, darkThr, 255, cv::THRESH_BINARY_INV);

        cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat connected;
        cv::morphologyEx(darkBin, connected, cv::MORPH_CLOSE, closeKernel);

        cv::Mat hKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 1));
        cv::Mat hLines;
        cv::morphologyEx(connected, hLines, cv::MORPH_OPEN, hKernel);

        cv::Mat vKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 40));
        cv::Mat vLines;
        cv::morphologyEx(connected, vLines, cv::MORPH_OPEN, vKernel);

        cv::Mat linesMask;
        cv::bitwise_or(hLines, vLines, linesMask);

        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(linesMask, linesMask, dilateKernel);

        cv::bitwise_or(contentMask, linesMask, contentMask);
    }

    // 保护彩色内容（印章、签名、红章）
    if (options.protectColor) {
        cv::Mat colorMask;
        cv::inRange(S, options.colorSatMin, 255, colorMask);
        cv::Mat valMask;
        cv::inRange(V, 40, 255, valMask);
        cv::bitwise_and(colorMask, valMask, colorMask);
        
        // 印章边缘膨胀保护，避免涂白生硬
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(colorMask, colorMask, dilateKernel);

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

    // 7. 背景白化
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

    // 将背景涂白，被保护的内容保留原样
    dst.setTo(white, bgMask);

    result.image = dst;
    result.whitenedPixels = cv::countNonZero(bgMask);
    result.ok = true;
    return result;
}

} // namespace process