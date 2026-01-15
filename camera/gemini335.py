import cv2
import numpy as np
from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBFrameAggregateOutputMode,
    OBFormat,
)


class Gemini335Camera:
    """
    Gemini 335 / 335L / 335Le 稳定版相机封装
    - 支持 RGB / Depth 单流模式
    - 符合 Orbbec 官方 Python SDK 使用规范
    """

    def __init__(self, launch_mode: str):
        """
        :param launch_mode: 'rgb' or 'depth'
        """
        if launch_mode not in ("rgb", "depth"):
            raise ValueError("launch_mode must be 'rgb' or 'depth'")

        self.launch_mode = launch_mode
        self.pipeline = Pipeline()
        self.config = Config()

        # ===============================
        # 1. 选择并启用 Stream Profile
        # ===============================
        if self.launch_mode == "rgb":
            profile_list = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            # color_profile = profile_list.get_default_video_stream_profile()
            # --- 修改：直接指定 1920x1080, MJPG, 30FPS ---
            color_profile = profile_list.get_video_stream_profile(
                1920, 1080, OBFormat.MJPG, 30
            )
            # # --- 打印确认信息 ---
            # print(f"----------------------------------------")
            # print(f"[Info] RGB Profile Selected:")
            # print(f"       Width : {color_profile.get_width()}")
            # print(f"       Height: {color_profile.get_height()}")
            # print(f"       FPS   : {color_profile.get_fps()}")
            # print(f"       Format: {color_profile.get_format()}")  # 应该是 MJPG (枚举值)
            # print(f"----------------------------------------")
            # -------------------------------------------

            self.config.enable_stream(color_profile)

        elif self.launch_mode == "depth":
            profile_list = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            depth_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)

        # ==========================================
        # 2. 必须设置：帧聚合输出模式（关键）
        # ==========================================
        self.config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )

    def start(self):
        """启动相机 Pipeline"""
        self.pipeline.start(self.config)

    def stop(self):
        """停止相机 Pipeline"""
        self.pipeline.stop()

    def get_frame(self):
        """
        获取一帧图像：
        - RGB 模式：返回 BGR uint8 图像（可直接给 OpenCV / YOLO）
        - Depth 模式：返回可视化后的深度图（uint8, colormap）
        """

        # 等待 FrameSet（官方推荐必须判空）
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None

        # ===============================
        # RGB 模式
        # ===============================
        if self.launch_mode == "rgb":
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None

            # Gemini 335 Color 默认是压缩流，必须 imdecode
            buffer = np.frombuffer(
                color_frame.get_data(), dtype=np.uint8
            )
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

            if image is None:
                return None

            return image

        # ===============================
        # Depth 模式
        # ===============================
        else:
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                return None

            depth_data = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            )

            height = depth_frame.get_height()
            width = depth_frame.get_width()

            depth = depth_data.reshape((height, width))

            # 深度可视化（仅用于显示）
            depth_vis = cv2.normalize(
                depth, None, 0, 255,
                cv2.NORM_MINMAX, cv2.CV_8U
            )
            depth_vis = cv2.applyColorMap(
                depth_vis, cv2.COLORMAP_JET
            )

            return depth_vis