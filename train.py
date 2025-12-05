from ultralytics import YOLO
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
import cv2
# ===================== 1. 定义路径 =====================
BASE_PATH = "F:/archive(11)/VOC2012"  # 需替换为实际VOC数据集路径
XML_PATH = os.path.join(BASE_PATH, "Annotations")
IMG_PATH = os.path.join(BASE_PATH, "JPEGImages")
YOLO_DATA_PATH = os.path.join(BASE_PATH, "yolo_format")  # YOLO格式数据集保存路径
os.makedirs(YOLO_DATA_PATH, exist_ok=True)
os.makedirs(os.path.join(YOLO_DATA_PATH, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(YOLO_DATA_PATH, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(YOLO_DATA_PATH, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(YOLO_DATA_PATH, "labels", "val"), exist_ok=True)


# ===================== 2. XML解析工具类 =====================
class XmlParser:
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.root = ET.parse(xml_file).getroot()
        self.objects = self.root.findall("object")
        self.img_path = os.path.join(IMG_PATH, self.root.find('filename').text)
        self.image_id = self.root.find("filename").text
        self.width = int(self.root.find("size/width").text)  # 新增：图像宽度
        self.height = int(self.root.find("size/height").text)  # 新增：图像高度

    def get_names(self):
        names = []
        for obj in self.objects:
            names.append(obj.find("name").text)
        return np.array(names)

    def get_boxes(self):
        boxes = []
        for obj in self.objects:
            bndbox = obj.find("bndbox")
            # 先转float再取整（处理浮点数标签）
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))  # 修复此处错误
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            boxes.append([xmin, ymin, xmax, ymax])
        return np.array(boxes)


# ===================== 3. VOC转YOLO格式（标签归一化） =====================
def voc_to_yolo(xml_files, train_ratio=0.8):
    # 收集所有类别
    all_classes = []
    for file in xml_files:
        parser = XmlParser(file)
        all_classes.extend(parser.get_names())
    all_classes = list(set(all_classes))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_classes)

    # 划分训练/验证集
    np.random.shuffle(xml_files)
    train_files = xml_files[:int(len(xml_files) * train_ratio)]
    val_files = xml_files[len(train_files):]

    # 生成YOLO格式标签和图像
    for phase, files in [("train", train_files), ("val", val_files)]:
        for file in files:
            parser = XmlParser(file)
            img = cv2.imread(parser.img_path)
            h, w = img.shape[:2]
            labels = []
            for name, box in zip(parser.get_names(), parser.get_boxes()):
                # VOC坐标转YOLO归一化坐标（x_center, y_center, width, height）
                xmin, ymin, xmax, ymax = box
                x_center = (xmin + xmax) / 2 / w
                y_center = (ymin + ymax) / 2 / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                class_id = label_encoder.transform([name])[0]
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 保存图像和标签
            img_save_path = os.path.join(YOLO_DATA_PATH, "images", phase, parser.image_id)
            label_save_path = os.path.join(YOLO_DATA_PATH, "labels", phase,
                                           os.path.splitext(parser.image_id)[0] + ".txt")
            cv2.imwrite(img_save_path, img)
            with open(label_save_path, "w") as f:
                f.write("\n".join(labels))

    # 生成YOLO配置文件（yaml）
    yaml_content = f"""
path: {YOLO_DATA_PATH}
train: images/train
val: images/val
names:
{chr(10).join([f"  {i}: {cls}" for i, cls in enumerate(label_encoder.classes_)])}
    """
    with open(os.path.join(YOLO_DATA_PATH, "voc2012.yaml"), "w") as f:
        f.write(yaml_content.strip())

    return os.path.join(YOLO_DATA_PATH, "voc2012.yaml")


# ===================== 4. 执行格式转换并训练YOLOv11 =====================



if __name__ == '__main__':
    # 获取所有XML文件
    XML_FILES = [os.path.join(XML_PATH, f) for f in os.listdir(XML_PATH) if f.endswith(".xml")]

    # 生成YOLO格式数据集和yaml配置
    yaml_path = voc_to_yolo(XML_FILES)
    model = YOLO("yolo11s.pt")
    model.train(data=yaml_path,
                epochs=150,
                batch=1,
                workers=0,
                imgsz=640)


    # 加载YOLOv11模型并训练
    # model = YOLO("yolo11s.pt")
    # model.train(data=r"E:/code/yolov11/mydata.yaml",
    #             epochs=150,
    #             batch=1,
    #             workers=0,
    #             imgsz=640)