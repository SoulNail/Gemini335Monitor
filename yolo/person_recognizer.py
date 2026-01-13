import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import json

class PersonRecognizer:
    def __init__(self, similarity_thresh=0.6):
        """
        similarity_thresh: 余弦相似度阈值
        """
        self.similarity_thresh = similarity_thresh

        # ✅ InsightFace：人脸检测 + 识别
        self.app = FaceAnalysis(
            name="buffalo_l",   # 官方推荐组合
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # ✅ 人脸特征库
        self.face_db = {}
        self.load_face_db("face_db/face_db.json")

        print("[INFO] InsightFace initialized")

    def load_face_db(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.face_db = {
            name: np.array(emb, dtype=np.float32)
            for name, emb in data.items()
        }

        print(f"[INFO] Loaded {len(self.face_db)} identities")

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def identify(self, embedding):
        best_name = "Unknown"
        best_score = 0.0

        for name, db_emb in self.face_db.items():
            score = self.cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.similarity_thresh:
            return best_name, best_score
        else:
            return "Unknown", best_score

    def process(self, frame, detections):
        """
        frame: 原始 BGR 图像
        detections: YOLO 返回的人框
        """

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # ✅ 在“人框”里做人脸检测
            faces = self.app.get(person_roi)

            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)

                # 映射回整帧坐标
                gx1, gy1 = x1 + fx1, y1 + fy1
                gx2, gy2 = x1 + fx2, y1 + fy2

                embedding = face.embedding
                name, score = self.identify(embedding)

                # ✅ 画人脸框
                cv2.rectangle(
                    frame, (gx1, gy1), (gx2, gy2),
                    (0, 0, 255), 2
                )

                # ✅ 标注身份
                cv2.putText(
                    frame,
                    f"{name} {score:.2f}",
                    (gx1, gy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        return frame