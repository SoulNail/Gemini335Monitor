import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis

FACE_IMG_DIR = "face_db/images"
OUTPUT_JSON = "face_db/face_db.json"

app = FaceAnalysis(name="buffalo_l",
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

face_db = {}

for person_name in os.listdir(FACE_IMG_DIR):
    person_dir = os.path.join(FACE_IMG_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"[WARN] No face in {img_path}")
            continue

        embeddings.append(faces[0].embedding)

    if embeddings:
        mean_emb = np.mean(embeddings, axis=0)
        face_db[person_name] = mean_emb.tolist()
        print(f"[OK] Registered {person_name}")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(face_db, f, indent=2)

print("Face DB saved to", OUTPUT_JSON)