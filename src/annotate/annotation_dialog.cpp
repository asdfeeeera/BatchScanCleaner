#include "annotation_dialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QInputDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QScrollBar>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QShortcut>
#include <QKeySequence>
#include <QScreen>
#include <QGuiApplication>
#include <QPen>
#include <QBrush>
#include <QColor>

namespace annotate {

const char *kCategoryWrongPage    = "wrong_page";
const char *kCategoryBleedThrough = "bleed_through";
const char *kCategoryBindingHole  = "binding_hole";
const char *kCategoryStain        = "stain";
const char *kCategoryCorrectPage  = "correct_page";

// ============================================================
// 构造
// ============================================================
AnnotationDialog::AnnotationDialog(const QStringList &imageFiles,
                                   const QString &imageRootDir,
                                   const QString &outputRootDir,
                                   QWidget *parent)
    : QDialog(parent)
    , m_imageFiles(imageFiles)
    , m_imageRootDir(imageRootDir)
    , m_outputRootDir(outputRootDir)
{
    setWindowTitle(QString::fromUtf8("文档异常标注工具"));
    resize(1400, 900);

    setupUi();

    if (!m_imageFiles.isEmpty()) {
        int startIdx = firstUnlabeledIndex();
        if (startIdx < 0) startIdx = 0;
        loadImageAt(startIdx);
    }
}

AnnotationDialog::~AnnotationDialog() = default;

// ============================================================
// UI
// ============================================================
void AnnotationDialog::setupUi()
{
    m_scene = new QGraphicsScene(this);
    m_view = new QGraphicsView(m_scene, this);
    m_view->setRenderHint(QPainter::SmoothPixmapTransform);
    m_view->setDragMode(QGraphicsView::NoDrag);
    m_view->setAlignment(Qt::AlignCenter);
    m_view->setBackgroundBrush(QBrush(QColor(60, 60, 60)));

    m_view->viewport()->installEventFilter(this);

    m_boxList = new QListWidget(this);
    connect(m_boxList, &QListWidget::itemSelectionChanged,
            this, &AnnotationDialog::onBoxListSelectionChanged);

    m_categoryCombo = new QComboBox(this);
    m_categoryCombo->addItem(QString::fromUtf8("错误页码"),
                             QString::fromUtf8(kCategoryWrongPage));
    m_categoryCombo->addItem(QString::fromUtf8("透印错码"),
                             QString::fromUtf8(kCategoryBleedThrough));
    m_categoryCombo->addItem(QString::fromUtf8("装订孔"),
                             QString::fromUtf8(kCategoryBindingHole));
    m_categoryCombo->addItem(QString::fromUtf8("顽固污渍"),
                             QString::fromUtf8(kCategoryStain));
    m_categoryCombo->addItem(QString::fromUtf8("正确页码"),
                             QString::fromUtf8(kCategoryCorrectPage));

    m_valueEdit = new QLineEdit(this);
    m_valueEdit->setPlaceholderText(
        QString::fromUtf8("错误页码请填写数字（其它类别可留空）"));

    m_applyBtn  = new QPushButton(QString::fromUtf8("应用修改"), this);
    m_deleteBtn = new QPushButton(QString::fromUtf8("删除选中框"), this);

    connect(m_applyBtn, &QPushButton::clicked,
            this, &AnnotationDialog::onApplyAttributes);
    connect(m_deleteBtn, &QPushButton::clicked,
            this, &AnnotationDialog::onDeleteSelectedBox);

    QLabel *boxListLabel = new QLabel(QString::fromUtf8("已标注框："), this);
    QLabel *catLabel = new QLabel(QString::fromUtf8("类别："), this);
    QLabel *valLabel = new QLabel(QString::fromUtf8("数字："), this);

    QVBoxLayout *rightLayout = new QVBoxLayout();
    rightLayout->addWidget(boxListLabel);
    rightLayout->addWidget(m_boxList, 1);
    rightLayout->addWidget(catLabel);
    rightLayout->addWidget(m_categoryCombo);
    rightLayout->addWidget(valLabel);
    rightLayout->addWidget(m_valueEdit);
    rightLayout->addWidget(m_applyBtn);
    rightLayout->addWidget(m_deleteBtn);

    QWidget *rightWidget = new QWidget(this);
    rightWidget->setLayout(rightLayout);
    rightWidget->setMaximumWidth(320);

    m_prevBtn          = new QPushButton(QString::fromUtf8("← 上一张"), this);
    m_nextBtn          = new QPushButton(QString::fromUtf8("下一张 →"), this);
    m_nextUnlabeledBtn = new QPushButton(QString::fromUtf8("跳到未标注"), this);
    m_saveBtn          = new QPushButton(QString::fromUtf8("保存当前"), this);

    connect(m_prevBtn, &QPushButton::clicked,
            this, &AnnotationDialog::onPrevImage);
    connect(m_nextBtn, &QPushButton::clicked,
            this, &AnnotationDialog::onNextImage);
    connect(m_nextUnlabeledBtn, &QPushButton::clicked,
            this, &AnnotationDialog::onNextUnlabeled);
    connect(m_saveBtn, &QPushButton::clicked,
            this, &AnnotationDialog::onSaveCurrent);

    QHBoxLayout *bottomLayout = new QHBoxLayout();
    bottomLayout->addWidget(m_prevBtn);
    bottomLayout->addWidget(m_nextBtn);
    bottomLayout->addWidget(m_nextUnlabeledBtn);
    bottomLayout->addStretch();
    bottomLayout->addWidget(m_saveBtn);

    m_imageNameLabel = new QLabel(QString::fromUtf8("（未加载）"), this);
    m_statusLabel = new QLabel(QString::fromUtf8("就绪"), this);

    QHBoxLayout *topLayout = new QHBoxLayout();
    topLayout->addWidget(m_imageNameLabel, 1);
    topLayout->addWidget(m_statusLabel);

    QHBoxLayout *centerLayout = new QHBoxLayout();
    centerLayout->addWidget(m_view, 1);
    centerLayout->addWidget(rightWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(topLayout);
    mainLayout->addLayout(centerLayout, 1);
    mainLayout->addLayout(bottomLayout);

    // ★ 快捷键
    //   空格 = 下一张
    {
        QShortcut *scNext = new QShortcut(QKeySequence(Qt::Key_Space), this);
        scNext->setContext(Qt::WindowShortcut);
        connect(scNext, &QShortcut::activated,
                this, &AnnotationDialog::onNextImage);
    }

    //   Q = 错误页码（索引 0）
    {
        QShortcut *sc = new QShortcut(QKeySequence(Qt::Key_Q), this);
        sc->setContext(Qt::WindowShortcut);
        connect(sc, &QShortcut::activated, this, [this]() {
            m_categoryCombo->setCurrentIndex(0);
        });
    }

    //   W = 透印错码（索引 1）
    {
        QShortcut *sc = new QShortcut(QKeySequence(Qt::Key_W), this);
        sc->setContext(Qt::WindowShortcut);
        connect(sc, &QShortcut::activated, this, [this]() {
            m_categoryCombo->setCurrentIndex(1);
        });
    }

    //   E = 装订孔（索引 2）
    {
        QShortcut *sc = new QShortcut(QKeySequence(Qt::Key_E), this);
        sc->setContext(Qt::WindowShortcut);
        connect(sc, &QShortcut::activated, this, [this]() {
            m_categoryCombo->setCurrentIndex(2);
        });
    }

    //   R = 顽固污渍（索引 3）
    {
        QShortcut *sc = new QShortcut(QKeySequence(Qt::Key_R), this);
        sc->setContext(Qt::WindowShortcut);
        connect(sc, &QShortcut::activated, this, [this]() {
            m_categoryCombo->setCurrentIndex(3);
        });
    }

    //   T = 正确页码（索引 4）
    {
        QShortcut *sc = new QShortcut(QKeySequence(Qt::Key_T), this);
        sc->setContext(Qt::WindowShortcut);
        connect(sc, &QShortcut::activated, this, [this]() {
            m_categoryCombo->setCurrentIndex(4);
        });
    }

    // ★ ESC = 重置视图
    {
        QShortcut *scReset = new QShortcut(QKeySequence(Qt::Key_Escape), this);
        scReset->setContext(Qt::WindowShortcut);
        connect(scReset, &QShortcut::activated,
                this, &AnnotationDialog::resetView);
    }
}

// ============================================================
// 重置视图
// ============================================================
void AnnotationDialog::resetView()
{
    if (!m_view) return;
    m_view->resetTransform();
    if (m_pixmapItem) {
        m_view->fitInView(m_scene->sceneRect(), Qt::KeepAspectRatio);
    }
}

// ============================================================
// 加载指定索引的图片
// ============================================================
void AnnotationDialog::loadImageAt(int index)
{
    if (index < 0 || index >= m_imageFiles.size()) return;

    if (m_currentIndex >= 0 && m_currentIndex != index) {
        saveToJson();
    }

    m_currentIndex = index;
    const QString imagePath = m_imageFiles.at(index);

    QImage img;
    if (!img.load(imagePath)) {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
            QString::fromUtf8("无法加载图片：\n%1").arg(imagePath));
        return;
    }

    m_scene->clear();
    m_boxItems.clear();
    m_tempRect = nullptr;
    m_drawing = false;
    m_panning = false;
    m_pixmapItem = nullptr;

    QPixmap pix = QPixmap::fromImage(img);
    m_pixmapItem = m_scene->addPixmap(pix);
    m_pixmapItem->setPos(0, 0);
    m_scene->setSceneRect(0, 0, pix.width(), pix.height());

    m_view->resetTransform();
    QTimer::singleShot(0, this, [this]() {
        m_view->fitInView(m_scene->sceneRect(), Qt::KeepAspectRatio);
    });

    m_currentBoxes.clear();
    loadFromJson(m_currentBoxes);
    for (const BoxData &b : m_currentBoxes) {
        addBoxToScene(b);
    }
    refreshBoxList();

    const QFileInfo fi(imagePath);
    m_imageNameLabel->setText(
        QString::fromUtf8("[%1 / %2]  %3  (%4 x %5)")
            .arg(index + 1).arg(m_imageFiles.size())
            .arg(fi.fileName())
            .arg(img.width()).arg(img.height()));
    updateStatusLabel();
}

void AnnotationDialog::clearBoxItems()
{
    for (QGraphicsRectItem *item : m_boxItems) {
        if (item) m_scene->removeItem(item);
        delete item;
    }
    m_boxItems.clear();
}

QGraphicsRectItem *AnnotationDialog::createBoxItem(const QRectF &rect,
                                                    bool selected) const
{
    QGraphicsRectItem *item = new QGraphicsRectItem(rect);
    QPen pen(selected ? QColor(0, 255, 0) : QColor(255, 0, 0));
    pen.setWidth(2);
    pen.setCosmetic(true);
    item->setPen(pen);
    item->setBrush(QBrush(QColor(255, 0, 0, 30)));
    return item;
}

void AnnotationDialog::addBoxToScene(const BoxData &data)
{
    QGraphicsRectItem *item = createBoxItem(data.rect, false);
    m_scene->addItem(item);
    m_boxItems.append(item);
}

void AnnotationDialog::refreshBoxList()
{
    m_boxList->clear();
    for (int i = 0; i < m_currentBoxes.size(); ++i) {
        const BoxData &b = m_currentBoxes[i];
        QString text = QString::fromUtf8("#%1  %2").arg(i + 1).arg(b.category);
        if (!b.value.isEmpty()) {
            text += QString::fromUtf8(" = %1").arg(b.value);
        }
        text += QString::fromUtf8("  [%1,%2 %3x%4]")
                    .arg(int(b.rect.x())).arg(int(b.rect.y()))
                    .arg(int(b.rect.width())).arg(int(b.rect.height()));
        m_boxList->addItem(text);
    }
}

void AnnotationDialog::updateStatusLabel()
{
    m_statusLabel->setText(
        QString::fromUtf8("本图框数：%1").arg(m_currentBoxes.size()));
}

bool AnnotationDialog::eventFilter(QObject *obj, QEvent *event)
{
    if (obj != m_view->viewport()) {
        return QDialog::eventFilter(obj, event);
    }

    if (event->type() == QEvent::Wheel) {
        if (!m_pixmapItem) return true;
        QWheelEvent *we = static_cast<QWheelEvent*>(event);
        const double factor = (we->angleDelta().y() > 0)
                              ? 1.15
                              : 1.0 / 1.15;
        m_view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        m_view->scale(factor, factor);
        m_view->setTransformationAnchor(QGraphicsView::AnchorViewCenter);
        return true;
    }

    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *me = static_cast<QMouseEvent*>(event);

        if (me->button() == Qt::MiddleButton) {
            m_panning = true;
            m_panStart = me->pos();
            m_view->viewport()->setCursor(Qt::ClosedHandCursor);
            return true;
        }

        if (me->button() == Qt::LeftButton && m_pixmapItem) {
            m_drawStart = m_view->mapToScene(me->pos());
            m_drawing = true;

            if (m_tempRect) {
                m_scene->removeItem(m_tempRect);
                delete m_tempRect;
                m_tempRect = nullptr;
            }
            m_tempRect = new QGraphicsRectItem(QRectF(m_drawStart, m_drawStart));
            QPen pen(QColor(0, 200, 255));
            pen.setWidth(2);
            pen.setCosmetic(true);
            m_tempRect->setPen(pen);
            m_scene->addItem(m_tempRect);
            return true;
        }
    }
    else if (event->type() == QEvent::MouseMove) {
        QMouseEvent *me = static_cast<QMouseEvent*>(event);

        if (m_panning) {
            const QPoint delta = me->pos() - m_panStart;
            m_panStart = me->pos();
            QScrollBar *hBar = m_view->horizontalScrollBar();
            QScrollBar *vBar = m_view->verticalScrollBar();
            hBar->setValue(hBar->value() - delta.x());
            vBar->setValue(vBar->value() - delta.y());
            return true;
        }

        if (m_drawing && m_tempRect) {
            const QPointF cur = m_view->mapToScene(me->pos());
            QRectF r(m_drawStart, cur);
            r = r.normalized();
            m_tempRect->setRect(r);
            return true;
        }
    }
    else if (event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent *me = static_cast<QMouseEvent*>(event);

        if (me->button() == Qt::MiddleButton && m_panning) {
            m_panning = false;
            m_view->viewport()->unsetCursor();
            return true;
        }

        if (me->button() == Qt::LeftButton && m_drawing) {
            m_drawing = false;

            if (m_tempRect) {
                const QRectF r = m_tempRect->rect();
                m_scene->removeItem(m_tempRect);
                delete m_tempRect;
                m_tempRect = nullptr;

                if (r.width() >= 5 && r.height() >= 5) {
                    BoxData b;
                    b.rect = r;
                    b.category = m_categoryCombo->currentData().toString();
                    b.value = m_valueEdit->text().trimmed();

                    const bool needNumber =
                        (b.category == QString::fromUtf8(kCategoryWrongPage) ||
                         b.category == QString::fromUtf8(kCategoryBleedThrough));

                    if (needNumber) {
                        QInputDialog dlg(this);
                        dlg.setWindowTitle(QString::fromUtf8("输入数字"));
                        dlg.setLabelText(QString::fromUtf8("数字："));
                        dlg.setTextValue(b.value);
                        dlg.setInputMode(QInputDialog::TextInput);

                        const QPoint viewPt = m_view->mapFromScene(r.bottomLeft());
                        const QPoint globalPt = m_view->viewport()->mapToGlobal(viewPt);
                        dlg.move(globalPt + QPoint(15, 15));

                        if (dlg.exec() == QDialog::Accepted) {
                            b.value = dlg.textValue().trimmed();
                        } else {
                            return true;
                        }
                    }

                    m_currentBoxes.append(b);
                    addBoxToScene(b);
                    refreshBoxList();
                    updateStatusLabel();
                }
            }
            return true;
        }
    }

    return QDialog::eventFilter(obj, event);
}

void AnnotationDialog::onPrevImage()
{
    if (m_currentIndex <= 0) return;
    loadImageAt(m_currentIndex - 1);
}

void AnnotationDialog::onNextImage()
{
    if (m_currentIndex >= m_imageFiles.size() - 1) return;
    loadImageAt(m_currentIndex + 1);
}

void AnnotationDialog::onNextUnlabeled()
{
    for (int i = m_currentIndex + 1; i < m_imageFiles.size(); ++i) {
        const QString jsonPath = jsonPathForImage(m_imageFiles.at(i));
        if (!QFile::exists(jsonPath)) {
            loadImageAt(i);
            return;
        }
    }
    QMessageBox::information(this, QString::fromUtf8("提示"),
        QString::fromUtf8("后面没有未标注的图片了。"));
}

void AnnotationDialog::onSaveCurrent()
{
    if (m_currentIndex < 0) return;
    if (saveToJson()) {
        m_statusLabel->setText(
            QString::fromUtf8("已保存：%1 个框").arg(m_currentBoxes.size()));
    } else {
        QMessageBox::warning(this, QString::fromUtf8("错误"),
            QString::fromUtf8("保存失败。"));
    }
}

void AnnotationDialog::onDeleteSelectedBox()
{
    const int row = m_boxList->currentRow();
    if (row < 0 || row >= m_currentBoxes.size()) return;

    m_currentBoxes.removeAt(row);

    if (row < m_boxItems.size()) {
        QGraphicsRectItem *item = m_boxItems.takeAt(row);
        if (item) {
            m_scene->removeItem(item);
            delete item;
        }
    }

    refreshBoxList();
    updateStatusLabel();
}

void AnnotationDialog::onBoxListSelectionChanged()
{
    const int row = m_boxList->currentRow();

    for (int i = 0; i < m_boxItems.size(); ++i) {
        QGraphicsRectItem *item = m_boxItems[i];
        if (!item) continue;

        QPen pen(i == row ? QColor(0, 255, 0) : QColor(255, 0, 0));
        pen.setWidth(2);
        pen.setCosmetic(true);
        item->setPen(pen);
    }

    if (row >= 0 && row < m_currentBoxes.size()) {
        const BoxData &b = m_currentBoxes[row];
        const int catIdx = m_categoryCombo->findData(b.category);
        if (catIdx >= 0) m_categoryCombo->setCurrentIndex(catIdx);
        m_valueEdit->setText(b.value);
    }
}

void AnnotationDialog::onApplyAttributes()
{
    const int row = m_boxList->currentRow();
    if (row < 0 || row >= m_currentBoxes.size()) return;

    BoxData &b = m_currentBoxes[row];
    b.category = m_categoryCombo->currentData().toString();
    b.value = m_valueEdit->text().trimmed();

    refreshBoxList();
    m_boxList->setCurrentRow(row);
}

QString AnnotationDialog::jsonPathForImage(const QString &imagePath) const
{
    const QFileInfo fi(imagePath);
    QDir rootDir(m_imageRootDir);
    const QString rel = rootDir.relativeFilePath(fi.absoluteFilePath());

    QString relJson = rel;
    const QString lower = relJson.toLower();
    if (lower.endsWith(QStringLiteral(".jpeg"))) {
        relJson.chop(5);
    } else if (lower.endsWith(QStringLiteral(".tiff"))) {
        relJson.chop(5);
    } else if (lower.endsWith(QStringLiteral(".jpg")) ||
               lower.endsWith(QStringLiteral(".png")) ||
               lower.endsWith(QStringLiteral(".bmp")) ||
               lower.endsWith(QStringLiteral(".tif"))) {
        relJson.chop(4);
    }
    relJson += QStringLiteral(".json");

    return QDir(m_outputRootDir).filePath(relJson);
}

bool AnnotationDialog::saveToJson() const
{
    if (m_currentIndex < 0 || m_currentIndex >= m_imageFiles.size()) return false;

    const QString imagePath = m_imageFiles.at(m_currentIndex);
    const QString jsonPath = jsonPathForImage(imagePath);

    const QString jsonDir = QFileInfo(jsonPath).absolutePath();
    if (!QDir().mkpath(jsonDir)) return false;

    if (m_currentBoxes.isEmpty()) {
        if (QFile::exists(jsonPath)) {
            QFile::remove(jsonPath);
        }
        return true;
    }

    QJsonArray arr;
    for (const BoxData &b : m_currentBoxes) {
        QJsonObject o;
        o.insert(QStringLiteral("x"), b.rect.x());
        o.insert(QStringLiteral("y"), b.rect.y());
        o.insert(QStringLiteral("w"), b.rect.width());
        o.insert(QStringLiteral("h"), b.rect.height());
        o.insert(QStringLiteral("category"), b.category);
        o.insert(QStringLiteral("value"), b.value);
        arr.append(o);
    }

    QJsonObject root;
    QDir rootDir(m_imageRootDir);
    const QFileInfo fi(imagePath);
    root.insert(QStringLiteral("image"),
                rootDir.relativeFilePath(fi.absoluteFilePath()));
    root.insert(QStringLiteral("boxes"), arr);

    QJsonDocument doc(root);

    QFile f(jsonPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    f.write(doc.toJson(QJsonDocument::Indented));
    f.close();
    return true;
}

bool AnnotationDialog::loadFromJson(QList<BoxData> &outBoxes) const
{
    outBoxes.clear();
    if (m_currentIndex < 0 || m_currentIndex >= m_imageFiles.size()) return false;

    const QString imagePath = m_imageFiles.at(m_currentIndex);
    const QString jsonPath = jsonPathForImage(imagePath);
    if (!QFile::exists(jsonPath)) return false;

    QFile f(jsonPath);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray raw = f.readAll();
    f.close();

    QJsonParseError err;
    const QJsonDocument doc = QJsonDocument::fromJson(raw, &err);
    if (err.error != QJsonParseError::NoError) return false;
    if (!doc.isObject()) return false;

    const QJsonObject root = doc.object();
    const QJsonArray arr = root.value(QStringLiteral("boxes")).toArray();

    for (const QJsonValue &v : arr) {
        const QJsonObject o = v.toObject();
        BoxData b;
        b.rect = QRectF(o.value(QStringLiteral("x")).toDouble(),
                        o.value(QStringLiteral("y")).toDouble(),
                        o.value(QStringLiteral("w")).toDouble(),
                        o.value(QStringLiteral("h")).toDouble());
        b.category = o.value(QStringLiteral("category")).toString();
        b.value = o.value(QStringLiteral("value")).toString();
        outBoxes.append(b);
    }
    return true;
}

int AnnotationDialog::firstUnlabeledIndex() const
{
    for (int i = 0; i < m_imageFiles.size(); ++i) {
        const QString jsonPath = jsonPathForImage(m_imageFiles.at(i));
        if (!QFile::exists(jsonPath)) return i;
    }
    return -1;
}

void AnnotationDialog::moveToCenter()
{
    QScreen *screen = QGuiApplication::primaryScreen();
    if (!screen) return;

    const QRect avail = screen->availableGeometry();
    const int x = avail.left() + (avail.width()  - width())  / 2;
    const int y = avail.top()  + (avail.height() - height()) / 2;
    move(qMax(avail.left(), x), qMax(avail.top(), y));
}

} // namespace annotate