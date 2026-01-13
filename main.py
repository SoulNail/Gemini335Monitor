import argparse
import cv2
import time

from camera.gemini335 import Gemini335Camera
from yolo.yolo_v8_person import YOLOv8PersonDetector
from yolo.person_recognizer import PersonRecognizer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini335 Camera Application"
    )

    parser.add_argument(
        "--launch",
        type=str,
        choices=["rgb", "depth"],
        required=True,
        help="Launch mode: rgb or depth"
    )

    parser.add_argument(
        "--detect",
        type=str2bool,
        default=False,
        help="Enable person detection (true / false)"
    )

    args = parser.parse_args()

    # ===============================
    # 初始化相机
    # ===============================
    camera = Gemini335Camera(args.launch)
    camera.start()

    # ===============================
    # 是否启用 YOLO
    # ===============================
    yolo = None
    if args.detect and args.launch == "rgb":
        print("[INFO] Person detection ENABLED")
        yolo = YOLOv8PersonDetector(
            weight_path="yolo/weights/yolov8s.pt",
            device="cuda"
        )
    else:
        print("[INFO] Person detection DISABLED")

    prev_time = time.time()

    print(f"[INFO] Gemini335 started in {args.launch.upper()} mode")
    print("[INFO] Application running... (Ctrl+C to exit)")

    person_recognizer = PersonRecognizer()
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            # ===============================
            # FPS 计算
            # ===============================
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # ===============================
            # 人员检测
            # ===============================
            if yolo is not None:
                frame, detections = yolo.process(frame)
                frame = person_recognizer.process(frame, detections)

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Gemini335", frame)

            # ⚠️ 仍然需要 waitKey，否则窗口无法刷新
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, exiting...")

    finally:
        camera.stop()
        # cv2.destroyAllWindows()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("[INFO] Application exited cleanly")


if __name__ == "__main__":
    main()