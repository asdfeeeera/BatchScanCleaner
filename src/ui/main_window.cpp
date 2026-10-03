void MainWindow::onProcessErrPage()
{
    if (!m_hasImage || m_currentMat.empty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请先打开一张图片。"));
        return;
    }

    if (m_currentImagePath.isEmpty()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("请通过打开图片或双击列表打开，才能解析文件名页码。"));
        return;
    }

    statusBar()->showMessage(QString::fromUtf8("正在处理错误页码..."));

    process::ErrPageOptions options;
    options.detectTopLeft          = true;
    options.topLeftWidthRatio      = 0.12;
    options.topLeftHeightRatio     = 0.08;
    options.detectTopRight         = true;
    options.topRightWidthRatio     = 0.25;
    options.topRightHeightRatio    = 0.15;
    options.detectBottomRight      = true;
    options.bottomRightWidthRatio  = 0.25;
    options.bottomRightHeightRatio = 0.15;
    options.detectBottomLeft       = false;
    options.minDigitHeight         = 20;
    options.maxDigitHeight         = 100;
    options.minDigitWidth          = 10;
    options.maxDigitWidth          = 150;
    options.crossLineRatio         = 0.5;
    options.fillWhite              = true;
    options.tesseractPath          = QString();

    const cv::Mat srcForThumb = m_currentMat.clone();

    const process::ErrPageResult result =
        process::ErrPage::process(m_currentMat, m_currentImagePath, options);

    if (!result.ok) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
                             QString::fromUtf8("错误页码处理失败。"));
        statusBar()->showMessage(QString::fromUtf8("错误页码处理失败"));
        return;
    }

    m_currentMat = result.image;
    showMatOnPreview(result.markedImage);

    // Step 1: clear old pending items for this file
    {
        const QList<analyze::PendingItem> allItems =
            analyze::PendingCenter::instance().allItems();
        QList<int> idsToDrop;
        for (const analyze::PendingItem &x : allItems) {
            if (x.sourceImagePath == m_currentImagePath) {
                idsToDrop.append(x.id);
            }
        }
        if (!idsToDrop.isEmpty()) {
            analyze::PendingCenter::instance().setDecisionByIds(
                idsToDrop, analyze::PendingDecision::Rejected);
            analyze::PendingCenter::instance().clearDecided();
        }
    }

    // Step 2: collect crossed boxes only (skip items matching correct page)
    std::vector<cv::Rect> crossedBoxes;
    for (const auto &it : result.items) {
        const bool matchesCorrect =
            (result.correctPage >= 0 &&
             it.recognizedNumber == result.correctPage);
        if (matchesCorrect) continue;
        if (!it.isCrossed) continue;

        crossedBoxes.push_back(it.boundingBox);
    }

    // Step 3: merge adjacent crossed boxes on the same row
    // (so "0","5","1" become one box "051")
    std::vector<cv::Rect> mergedBoxes;
    if (!crossedBoxes.empty()) {
        std::sort(crossedBoxes.begin(), crossedBoxes.end(),
                  [](const cv::Rect &a, const cv::Rect &b) {
                      if (std::abs(a.y - b.y) < a.height / 2) return a.x < b.x;
                      return a.y < b.y;
                  });

        cv::Rect current = crossedBoxes[0];
        for (size_t i = 1; i < crossedBoxes.size(); ++i) {
            const cv::Rect &next = crossedBoxes[i];

            const int sameRow = std::abs(current.y - next.y) < current.height / 2;
            const int gap = next.x - (current.x + current.width);
            const bool closeGap = gap >= 0 && gap < current.height * 2;

            if (sameRow && closeGap) {
                const int x1 = std::min(current.x, next.x);
                const int y1 = std::min(current.y, next.y);
                const int x2 = std::max(current.x + current.width,
                                         next.x + next.width);
                const int y2 = std::max(current.y + current.height,
                                         next.y + next.height);
                current = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            } else {
                mergedBoxes.push_back(current);
                current = next;
            }
        }
        mergedBoxes.push_back(current);
    }

    // Step 4: add one pending item per merged box
    const QFileInfo fi(m_currentImagePath);
    int addedCount = 0;
    for (const cv::Rect &box : mergedBoxes) {
        analyze::PendingItem p;
        p.type = analyze::PendingType::WrongPageNumber;
        p.suggestedAction = analyze::PendingAction::Remove;
        p.sourceImagePath = m_currentImagePath;
        p.fileName = fi.fileName();
        p.boundingBox = box;
        p.confidence = 0.9;
        p.reason = QString::fromUtf8("检测到划线数字，与正确页码不符");
        p.detail = QString::fromUtf8("划线区域，正确页码：%1")
                       .arg(result.correctPage);

        cv::Rect safe = box &
                        cv::Rect(0, 0, srcForThumb.cols, srcForThumb.rows);
        if (safe.width > 0 && safe.height > 0) {
            p.thumbnail = srcForThumb(safe).clone();
        }

        analyze::PendingCenter::instance().addItem(p);
        ++addedCount;
    }

    QString pageInfo;
    if (result.correctPage >= 0) {
        pageInfo = QString::fromUtf8("正确页码 %1").arg(result.correctPage);
    } else {
        pageInfo = QString::fromUtf8("文件名无页码");
    }

    QString msg;
    if (result.skipped) {
        msg = QString::fromUtf8("错误页码处理：角落未检测到数字。%1")
                  .arg(pageInfo);
    } else {
        msg = QString::fromUtf8("错误页码处理完成：%1，检测到 %2 个数字块，自动删除划线 %3 个，待确认 %4 个")
                  .arg(pageInfo)
                  .arg(static_cast<int>(result.items.size()))
                  .arg(result.crossedRemoved)
                  .arg(result.pendingCount);
    }
    if (addedCount > 0) {
        msg += QString::fromUtf8("（已加入待确认中心 %1 项）").arg(addedCount);
    }

    statusBar()->showMessage(msg);
}