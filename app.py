import sys
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPen, QColor, QPixmap, QBrush, QFont

# 第三方美化库
from qt_material import apply_stylesheet
import qtawesome as qta

class BoxInfoDialog(QtWidgets.QDialog):
    def __init__(self, parent, label, conf, rect):
        super().__init__(parent)
        self.setWindowTitle(f"{label} 详情")
        self.setMinimumWidth(300)
        
        # 简约的弹窗样式
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        title = QtWidgets.QLabel(f"{label}")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)
        
        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(10)
        
        self.add_row(grid, 0, "置信度", f"{conf:.2%}")
        self.add_row(grid, 1, "坐标区域", f"[{int(rect.x())}, {int(rect.y())}, {int(rect.width())}x{int(rect.height())}]")

        layout.addWidget(title)
        layout.addLayout(grid)
        
        btn = QtWidgets.QPushButton("关闭")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def add_row(self, layout, row, key, value):
        k_label = QtWidgets.QLabel(key)
        k_label.setStyleSheet("color: #7f8c8d; font-weight: 600;")
        v_label = QtWidgets.QLabel(value)
        v_label.setStyleSheet("color: #34495e; font-family: 'Consolas', monospace;")
        layout.addWidget(k_label, row, 0)
        layout.addWidget(v_label, row, 1, alignment=Qt.AlignRight)

class DetectedBox(QGraphicsRectItem):
    def __init__(self, rect, label, conf, color, info_callback=None):
        super().__init__(rect)
        self.label = label
        self.conf = conf
        self.info_callback = info_callback
        
        # 设置半透明填充
        c = QColor(color)
        c.setAlpha(40) 
        self.setBrush(QBrush(c))
        
        pen = QPen(QColor(color), 2.5)
        pen.setJoinStyle(Qt.RoundJoin)
        self.setPen(pen)
        
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(2)

        # 标签背景
        self.text_bg = QGraphicsRectItem()
        self.text_bg.setBrush(QBrush(QColor(color)))
        
        # --- 修复点在这里 ---
        # 必须使用 QPen(Qt.NoPen)，不能直接用 Qt.NoPen
        self.text_bg.setPen(QPen(Qt.NoPen)) 
        # -------------------
        
        self.text_bg.setZValue(3)
        
        # 标签文字
        self.textItem = QGraphicsTextItem(f'{label} {conf:.2f}')
        self.textItem.setDefaultTextColor(Qt.white)
        self.textItem.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.textItem.setZValue(4)
        
        self.update_text_pos(rect)

    def update_text_pos(self, rect):
        self.textItem.setPos(rect.x(), rect.y() - 22)
        br = self.textItem.boundingRect()
        self.text_bg.setRect(rect.x(), rect.y() - 22, br.width(), 22)

    def hoverEnterEvent(self, event):
        c = self.brush().color()
        c.setAlpha(90)
        self.setBrush(c)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        c = self.brush().color()
        c.setAlpha(40)
        self.setBrush(c)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if self.info_callback:
            self.info_callback(self.label, self.conf, self.rect())
        super().mousePressEvent(event)

    def update_text_pos(self, rect):
        # 简单的标签定位逻辑
        self.textItem.setPos(rect.x(), rect.y() - 22)
        br = self.textItem.boundingRect()
        self.text_bg.setRect(rect.x(), rect.y() - 22, br.width(), 22)

    def hoverEnterEvent(self, event):
        # 悬停高亮
        c = self.brush().color()
        c.setAlpha(90)
        self.setBrush(c)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        c = self.brush().color()
        c.setAlpha(40)
        self.setBrush(c)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if self.info_callback:
            self.info_callback(self.label, self.conf, self.rect())
        super().mousePressEvent(event)

class ImageCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.imageItem = None
        self.box_items = []
        self.text_bg_items = [] 
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # 清新的画布背景
        self.setStyleSheet("background-color: #f8f9fa; border: none; border-radius: 15px;")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def set_image(self, pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        qtimg = QtGui.QImage.fromData(buf.getvalue())
        pix = QPixmap.fromImage(qtimg)
        
        self.scene.clear()
        self.box_items.clear()
        self.text_bg_items.clear()
        
        self.imageItem = QGraphicsPixmapItem(pix)
        self.imageItem.setZValue(1)
        
        # 添加阴影让图片看起来是浮在画布上
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.imageItem.setGraphicsEffect(shadow)
        
        self.scene.addItem(self.imageItem)
        self.setSceneRect(self.imageItem.boundingRect())
        self.fitInView(self.imageItem, Qt.KeepAspectRatio)

    def add_box(self, box, label, conf, color, info_callback):
        item = DetectedBox(box, label, conf, color, info_callback)
        self.scene.addItem(item)
        self.scene.addItem(item.text_bg)
        self.scene.addItem(item.textItem)
        self.box_items.append(item)
        self.text_bg_items.append(item.text_bg) # 追踪背景以便过滤
        return item

    def clear_boxes(self):
        # 重新加载图片时清理旧框
        if self.imageItem:
            # 只保留图片
            for item in self.scene.items():
                if item != self.imageItem:
                    self.scene.removeItem(item)
        self.box_items.clear()
        self.text_bg_items.clear()

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 0.85
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

class StatPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        
        header_layout = QtWidgets.QHBoxLayout()
        icon = QtWidgets.QLabel()
        icon.setPixmap(qta.icon('fa5s.chart-pie', color='#009688').pixmap(24, 24))
        
        self.label = QtWidgets.QLabel("检测统计")
        self.label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        
        header_layout.addWidget(icon)
        header_layout.addWidget(self.label)
        header_layout.addStretch()
        self.vbox.addLayout(header_layout)
        
        self.stats_container = QtWidgets.QWidget()
        self.stats_layout = QtWidgets.QVBoxLayout(self.stats_container)
        self.stats_layout.setSpacing(10)
        self.vbox.addWidget(self.stats_container)
        self.vbox.addStretch(1)

    def set_stats(self, stats_dict):
        # 清除旧统计
        while self.stats_layout.count():
            child = self.stats_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not stats_dict:
            empty = QtWidgets.QLabel("暂无数据")
            empty.setStyleSheet("color: #bdc3c7; font-style: italic; margin-top: 10px;")
            empty.setAlignment(Qt.AlignCenter)
            self.stats_layout.addWidget(empty)
            return

        for cls, num in stats_dict.items():
            row = QtWidgets.QWidget()
            row.setStyleSheet("background: #f8f9fa; border-radius: 6px;")
            r_layout = QtWidgets.QHBoxLayout(row)
            r_layout.setContentsMargins(10, 8, 10, 8)
            
            lbl_cls = QtWidgets.QLabel(cls)
            lbl_cls.setStyleSheet("font-weight: 600; color: #57606f;")
            
            lbl_num = QtWidgets.QLabel(str(num))
            lbl_num.setStyleSheet("font-weight: bold; color: #009688; background: #e0f2f1; padding: 2px 8px; border-radius: 10px;")
            
            r_layout.addWidget(lbl_cls)
            r_layout.addStretch()
            r_layout.addWidget(lbl_num)
            
            self.stats_layout.addWidget(row)

class ObjectDetectionApp(QtWidgets.QMainWindow):
    # 莫兰迪色系 / 清新色系
    COLORS = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD", 
        "#D4A5A5", "#9B59B6", "#3498DB", "#E67E22", "#2ECC71"
    ]

    def __init__(self):
        super().__init__()
        self.img_width = 820
        self.img_height = 640
        self.input_image = None
        self.result_boxes = []
        self.model = YOLO("./best.pt")
        
        self.init_ui()
        self.setWindowTitle("Vision AI 识别系统")
        self.resize(1280, 800)

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 左侧：画布
        self.canvas = ImageCanvas()
        
        # 给画布一个容器，增加一点边距或阴影效果
        canvas_container = QtWidgets.QWidget()
        canvas_layout = QtWidgets.QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0,0,0,0)
        canvas_layout.addWidget(self.canvas)
        
        # 右侧：控制面板
        right_panel = QtWidgets.QFrame()
        right_panel.setObjectName("RightPanel")
        right_panel.setFixedWidth(380)
        # 卡片式设计 CSS
        right_panel.setStyleSheet("""
            #RightPanel {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
            }
        """)
        # 添加阴影
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 20))
        right_panel.setGraphicsEffect(shadow)

        panel_layout = QtWidgets.QVBoxLayout(right_panel)
        panel_layout.setContentsMargins(30, 40, 30, 40)
        panel_layout.setSpacing(25)

        # 标题
        title_box = QtWidgets.QHBoxLayout()
        title_icon = QtWidgets.QLabel()
        title_icon.setPixmap(qta.icon('fa5s.eye', color='#009688').pixmap(32, 32))
        title_lbl = QtWidgets.QLabel("智能识别")
        title_lbl.setStyleSheet("font-size: 26px; font-weight: 900; color: #2c3e50; font-family: 'Microsoft YaHei UI';")
        title_box.addWidget(title_icon)
        title_box.addWidget(title_lbl)
        title_box.addStretch()
        panel_layout.addLayout(title_box)
        
        panel_layout.addSpacing(10)

        # 按钮组
        self.choose_btn = QtWidgets.QPushButton(" 选择图片")
        self.choose_btn.setIcon(qta.icon('fa5s.image', color='white'))
        self.choose_btn.setMinimumHeight(50)
        # 注意：qt-material 会接管按钮样式，我们通过 setProperty 来应用特定的类样式（如果有的话），或者仅依靠 theme
        
        self.infer_btn = QtWidgets.QPushButton(" 开始检测")
        self.infer_btn.setIcon(qta.icon('fa5s.magic', color='white'))
        self.infer_btn.setMinimumHeight(50)
        self.infer_btn.setEnabled(False)
        # 给检测按钮一个强调色样式 (qt-material 特性)
        self.infer_btn.setProperty('class', 'success') 

        panel_layout.addWidget(self.choose_btn)
        panel_layout.addWidget(self.infer_btn)
        
        # 分隔线
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("background: #ecf0f1;")
        panel_layout.addWidget(line)

        # 过滤器
        filter_layout = QtWidgets.QVBoxLayout()
        f_label = QtWidgets.QLabel("类别筛选")
        f_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #7f8c8d;")
        
        self.cls_filter = QtWidgets.QComboBox()
        self.cls_filter.setMinimumHeight(40)
        self.cls_filter.addItem("显示全部")
        
        filter_layout.addWidget(f_label)
        filter_layout.addWidget(self.cls_filter)
        panel_layout.addLayout(filter_layout)

        # 统计面板
        self.statpanel = StatPanel()
        panel_layout.addWidget(self.statpanel)
        
        panel_layout.addStretch()

        # 组装主布局
        main_layout.addWidget(canvas_container, 1)
        main_layout.addWidget(right_panel)

        # 信号连接
        self.choose_btn.clicked.connect(self.choose_image)
        self.infer_btn.clicked.connect(self.do_infer)
        self.cls_filter.currentTextChanged.connect(self.filter_changed)

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "",
                                                        "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if path:
            img = Image.open(path).convert("RGB")
            # 保持宽高比resize或者直接使用原图，这里为了性能限制一下最大边
            img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            self.input_image = img
            
            self.canvas.set_image(img)
            self.canvas.clear_boxes()
            
            self.result_boxes.clear()
            self.cls_filter.clear()
            self.cls_filter.addItem("显示全部")
            self.statpanel.set_stats({})
            
            self.infer_btn.setEnabled(True)

    def do_infer(self):
        if self.input_image is None:
            return
        
        # 界面反馈：更改鼠标状态
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        self.infer_btn.setText(" 检测中...")
        self.infer_btn.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        try:
            img_np = np.array(self.input_image)
            results = self.model.predict(img_np, imgsz=640, stream=True)
            
            self.canvas.clear_boxes()
            self.result_boxes.clear()
            
            labelset = set()
            stats = {}
            
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = [float(coord) for coord in xyxy]
                    
                    label_idx = int(box.cls[0].cpu().numpy())
                    label = result.names[label_idx]
                    conf = float(box.conf[0].cpu().numpy())
                    
                    labelset.add(label)
                    stats[label] = stats.get(label, 0) + 1
                    
                    rect = QRectF(x1, y1, x2-x1, y2-y1)
                    color = self.COLORS[label_idx % len(self.COLORS)]
                    
                    box_item = self.canvas.add_box(rect, label, conf, color, self.on_box_info)
                    self.result_boxes.append(box_item)
            
            self.statpanel.set_stats(stats)
            
            # 更新下拉框
            current_filter = self.cls_filter.currentText()
            self.cls_filter.blockSignals(True)
            self.cls_filter.clear()
            self.cls_filter.addItem("显示全部")
            for cls in sorted(labelset):
                self.cls_filter.addItem(cls)
            self.cls_filter.setCurrentText(current_filter if current_filter in labelset else "显示全部")
            self.cls_filter.blockSignals(False)
            
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.infer_btn.setText(" 开始检测")
            self.infer_btn.setEnabled(True)
            self.infer_btn.setIcon(qta.icon('fa5s.check', color='white')) # 完成后图标变化

    def filter_changed(self, txt):
        for box in self.result_boxes:
            visible = (txt == "显示全部" or box.label == txt)
            box.setVisible(visible)
            box.textItem.setVisible(visible)
            box.text_bg.setVisible(visible)

    def on_box_info(self, label, conf, rect):
        dlg = BoxInfoDialog(self, label, conf, rect)
        dlg.exec_()

if __name__ == '__main__':
    # 解决高分屏缩放问题
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    
    app = QtWidgets.QApplication(sys.argv)
    
    # 应用 qt_material 主题
    # 推荐使用 'light_teal.xml' 或 'light_blue.xml' 获得清新的效果
    # invert_secondary=True 可以让次要控件颜色反转，增加对比度
    apply_stylesheet(app, theme='light_teal.xml', invert_secondary=True)
    
    # 全局微调：增加字体大小，调整字体族
    app.setStyleSheet(app.styleSheet() + """
        * { font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif; }
        QPushButton { font-weight: bold; font-size: 15px; border-radius: 8px; }
        QComboBox { border-radius: 8px; padding: 5px; }
    """)

    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())