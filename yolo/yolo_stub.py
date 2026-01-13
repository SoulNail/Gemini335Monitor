import cv2
import numpy as np


class YOLOStub:
    def __init__(self):
        # 后续可在这里加载 torch / yolo 模型
        pass

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        YOLO 算法处理接口（占位）
        """
        h, w, _ = image_bgr.shape
        cv2.rectangle(image_bgr, (50, 50), (w - 50, h - 50), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            "YOLO Enabled (Stub)",
            (50, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return image_bgr