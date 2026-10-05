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
extern const char *kCategoryWrongPage;   // "wrong_page"
extern const char *kCategoryBindingHole; // "binding_hole"
extern const char *kCategoryStain;       // "stain"

// ============================================================
// 通用文档异常标注窗口
//   第一版只做"错误页码"标注；后续扩展装订孔/污渍
// ============================================================
class AnnotationDialog : public QDialog
{
    Q_OBJECT

public:
    // imageFiles    待标注的图片完整路径列表
    // imageRootDir  这些图片的根目录（用于计算相对路径，决定 JSON 存哪）
    // outputRootDir 标注结果存放根目录（如 E:/标注数据）
    explicit AnnotationDialog(const QStringList &imageFiles,
                              const QString &imageRootDir,
                              const QString &outputRootDir,
                              QWidget *parent = nullptr);
    ~AnnotationDialog() override;

    // 父窗口调用：把窗口移到屏幕中央
    void moveToCenter();

protected:
    // 给 view->viewport() 装事件过滤器，处理鼠标画框
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

    // ---------- 数据 ----------
    QStringList      m_imageFiles;      // 待标注图片
    QString          m_imageRootDir;    // 图片根目录
    QString          m_outputRootDir;   // 标注结果根目录
    int              m_currentIndex = -1;
    QList<BoxData>   m_currentBoxes;    // 当前图的标注框

    // ---------- 画框状态 ----------
    bool              m_drawing = false;
    QPointF           m_drawStart;
    QGraphicsRectItem *m_tempRect = nullptr;

    // ---------- 图形视图 ----------
    QGraphicsScene      *m_scene = nullptr;
    QGraphicsView       *m_view = nullptr;
    QGraphicsPixmapItem *m_pixmapItem = nullptr;
    QList<QGraphicsRectItem*> m_boxItems;   // 已确认的框（和 m_currentBoxes 一一对应）

    // ---------- 右侧面板 ----------
    QListWidget *m_boxList = nullptr;
    QComboBox   *m_categoryCombo = nullptr;
    QLineEdit   *m_valueEdit = nullptr;
    QPushButton *m_applyBtn = nullptr;
    QPushButton *m_deleteBtn = nullptr;

    // ---------- 底部按钮 ----------
    QPushButton *m_prevBtn = nullptr;
    QPushButton *m_nextBtn = nullptr;
    QPushButton *m_nextUnlabeledBtn = nullptr;
    QPushButton *m_saveBtn = nullptr;

    // ---------- 状态 ----------
    QLabel *m_imageNameLabel = nullptr;
    QLabel *m_statusLabel = nullptr;
};

} // namespace annotate