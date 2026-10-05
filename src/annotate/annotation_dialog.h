#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>
#include <QList>
#include <QRectF>
#include <QPointF>

class QGraphicsScene;
class QGraphicsView;
class QGraphicsPixmapItem;
class QGraphicsRectItem;
class QListWidget;
class QListWidgetItem;
class QPushButton;
class QLabel;
class QLineEdit;
class QComboBox;

namespace annotate {

// ============================================================
// 标注框数据
// ============================================================
struct BoxData
{
    QRectF  rect;         // 图片像素坐标系下的矩形 (x, y, w, h)
    QString category;     // 类别：见下面 category 常量
    QString value;        // 若 category == kCategoryWrongPage，这里存识别数字
};

// 类别常量（存到 JSON 里的字符串）
extern const char *kCategoryWrongPage;    // "wrong_page"     错误页码
extern const char *kCategoryBleedThrough; // "bleed_through"  透印错码
extern const char *kCategoryBindingHole;  // "binding_hole"   装订孔
extern const char *kCategoryStain;        // "stain"          顽固污渍

// ============================================================
// 通用文档异常标注窗口
// ============================================================
class AnnotationDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AnnotationDialog(const QStringList &imageFiles,
                              const QString &imageRootDir,
                              const QString &outputRootDir,
                              QWidget *parent = nullptr);
    ~AnnotationDialog() override;

    void moveToCenter();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void onPrevImage();
    void onNextImage();
    void onNextUnlabeled();
    void onSaveCurrent();
    void onDeleteSelectedBox();
    void onBoxListSelectionChanged();
    void onApplyAttributes();

private:
    void setupUi();
    void loadImageAt(int index);
    void clearBoxItems();
    void addBoxToScene(const BoxData &data);
    void refreshBoxList();
    void updateStatusLabel();
    bool saveToJson() const;
    bool loadFromJson(QList<BoxData> &outBoxes) const;
    QString jsonPathForImage(const QString &imagePath) const;
    int firstUnlabeledIndex() const;
    QGraphicsRectItem *createBoxItem(const QRectF &rect, bool selected) const;

    QStringList      m_imageFiles;
    QString          m_imageRootDir;
    QString          m_outputRootDir;
    int              m_currentIndex = -1;
    QList<BoxData>   m_currentBoxes;

    bool              m_drawing = false;
    QPointF           m_drawStart;
    QGraphicsRectItem *m_tempRect = nullptr;

    QGraphicsScene      *m_scene = nullptr;
    QGraphicsView       *m_view = nullptr;
    QGraphicsPixmapItem *m_pixmapItem = nullptr;
    QList<QGraphicsRectItem*> m_boxItems;

    QListWidget *m_boxList = nullptr;
    QComboBox   *m_categoryCombo = nullptr;
    QLineEdit   *m_valueEdit = nullptr;
    QPushButton *m_applyBtn = nullptr;
    QPushButton *m_deleteBtn = nullptr;

    QPushButton *m_prevBtn = nullptr;
    QPushButton *m_nextBtn = nullptr;
    QPushButton *m_nextUnlabeledBtn = nullptr;
    QPushButton *m_saveBtn = nullptr;

    QLabel *m_imageNameLabel = nullptr;
    QLabel *m_statusLabel = nullptr;
};

} // namespace annotate