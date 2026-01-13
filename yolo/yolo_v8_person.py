import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLOv8PersonDetector:
    def __init__(
        self,
        weight_path: str = "yolo/weights/yolov8s.pt",
        device: str = "cuda"
    ):
        """
        YOLOv8 人员检测器

        :param weight_path: yolov8s.pt 路径
        :param device: cuda or cpu
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weight_path)
        self.model.to(self.device)

        # COCO 中 person 的 class id = 0
        self.person_class_id = 0

        print(f"[YOLO] Loaded YOLOv8 model on {self.device}")

    def process(self, image_bgr):
        detections = []

        results = self.model(
            image_bgr,
            imgsz=640,
            conf=0.4,
            classes=[0],  # ✅ 只检测 person
            verbose=False
        )

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "score": float(score)
                })

                # 画人框（检测层）
                cv2.rectangle(
                    image_bgr, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

        return image_bgr, detections