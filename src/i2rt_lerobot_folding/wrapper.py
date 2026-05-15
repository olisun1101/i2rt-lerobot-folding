from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import cv2

from i2rt.robots.get_robot import get_yam_robot

from lerobot.robots import Robot, RobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode


def preprocess_rgb(img, size=(224, 224)):
    """
    Input expected:
        RGB image, HWC, uint8

    Output:
        CHW, uint8
    """
    if img is None:
        raise RuntimeError("RGB frame is None")

    img = cv2.resize(img, size)
    img = img.transpose(2, 0, 1)
    return img.astype(np.uint8)


def preprocess_depth(depth, size=(224, 224)):
    """
    Input expected:
        depth image, HW

    Output:
        1HW, float32
    """
    if depth is None:
        raise RuntimeError("Depth frame is None")

    depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)

    if depth.ndim == 2:
        depth = depth[None, :, :]

    return depth.astype(np.float32)


@RobotConfig.register_subclass("i2rt_dual_arm")
@dataclass
class I2RTDualArmConfig(RobotConfig):
    
    left_channel: str = "can_follower_l"
    right_channel: str = "can_follower_r"

    left_gripper_type: str = "linear_4310"
    right_gripper_type: str = "linear_4310"


    realsense_top_serial: str = "PUT_TOP_SERIAL_HERE"
    realsense_left_serial: str = "PUT_LEFT_SERIAL_HERE"
    realsense_right_serial: str = "PUT_RIGHT_SERIAL_HERE"

    camera_fps: int = 30

    # RealSense capture resolution
    image_width: int = 640
    image_height: int = 480

    # Policy input resolution
    resize_width: int = 224
    resize_height: int = 224

    use_depth: bool = True

    # 6 if no gripper, 7 if arm joints + gripper
    action_dim_per_arm: int = 7


class I2RTDualArmRobot(Robot):
    config_class = I2RTDualArmConfig
    name = "i2rt_dual_arm"

    def __init__(self, config: I2RTDualArmConfig):
        super().__init__(config)

        self.left_arm = None
        self.right_arm = None

        self.cam_top = None
        self.cam_left = None
        self.cam_right = None

    def _make_realsense(self, serial: str):
        cam_config = RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=self.config.camera_fps,
            width=self.config.image_width,
            height=self.config.image_height,
            color_mode=ColorMode.RGB,
            use_depth=self.config.use_depth,
        )
        return RealSenseCamera(cam_config)

    
    def connect(self):
        print("Connecting I2RT follower dual arm...")

        self.left_arm = get_yam_robot(
            channel=self.config.left_channel,
            gripper_type=self.config.left_gripper_type,
        )

        self.right_arm = get_yam_robot(
            channel=self.config.right_channel,
            gripper_type=self.config.right_gripper_type,
        )

        print("Opening RealSense cameras...")

        self.cam_top = self._make_realsense(self.config.realsense_top_serial)
        self.cam_left = self._make_realsense(self.config.realsense_left_serial)
        self.cam_right = self._make_realsense(self.config.realsense_right_serial)

        self.cam_top.connect()
        self.cam_left.connect()
        self.cam_right.connect()

        print("I2RT follower dual arm + RealSense cameras connected.")


    def disconnect(self):
        print("Disconnecting...")

        for cam in [self.cam_top, self.cam_left, self.cam_right]:
            if cam is not None:
                cam.disconnect()

        self.cam_top = None
        self.cam_left = None
        self.cam_right = None

        self.left_arm = None
        self.right_arm = None

        print("Disconnected.")

    @property
    def is_connected(self):
        return (
            self.left_arm is not None
            and self.right_arm is not None
            and self.cam_top is not None
            and self.cam_left is not None
            and self.cam_right is not None
        )

    @property
    def observation_features(self):
        d = self.config.action_dim_per_arm
        h = self.config.resize_height
        w = self.config.resize_width

        features = {
            "left_joint_pos": (d,),
            "right_joint_pos": (d,),

            "image_top": (3, h, w),
            "image_left": (3, h, w),
            "image_right": (3, h, w),
        }

        if self.config.use_depth:
            features.update({
                "depth_top": (1, h, w),
                "depth_left": (1, h, w),
                "depth_right": (1, h, w),
            })

        return features

    @property
    def action_features(self):
        d = self.config.action_dim_per_arm

        return {
            "left_joint_pos": (d,),
            "right_joint_pos": (d,),
        }

    def _read_realsense(self, cam, name: str):
        """
        Different LeRobot versions may return:
            - RGB image directly
            - dict with image/depth
        So inspect once if needed.
        """
        frame = cam.async_read(timeout_ms=200)

        # Case 1: frame is dict-like
        if isinstance(frame, dict):
            rgb = frame.get("image", None)
            if rgb is None:
                rgb = frame.get("color", None)
            depth = frame.get("depth", None)

        # Case 2: frame is just RGB image
        else:
            rgb = frame
            depth = None

        if rgb is None:
            raise RuntimeError(f"Failed to read RGB from {name} RealSense camera")

        if self.config.use_depth and depth is None:
            raise RuntimeError(
                f"Depth is enabled but failed to read depth from {name} RealSense camera"
            )

        return rgb, depth

    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        obs_l = self.left_arm.get_observations()
        obs_r = self.right_arm.get_observations()

        rgb_top, depth_top = self._read_realsense(self.cam_top, "top")
        rgb_left, depth_left = self._read_realsense(self.cam_left, "left")
        rgb_right, depth_right = self._read_realsense(self.cam_right, "right")

        size = (self.config.resize_width, self.config.resize_height)

        obs = {
            "left_joint_pos": np.asarray(obs_l["joint_pos"], dtype=np.float32),
            "right_joint_pos": np.asarray(obs_r["joint_pos"], dtype=np.float32),

            "image_top": preprocess_rgb(rgb_top, size=size),
            "image_left": preprocess_rgb(rgb_left, size=size),
            "image_right": preprocess_rgb(rgb_right, size=size),
        }

        if self.config.use_depth:
            obs.update({
                "depth_top": preprocess_depth(depth_top, size=size),
                "depth_left": preprocess_depth(depth_left, size=size),
                "depth_right": preprocess_depth(depth_right, size=size),
            })

        return obs

    def send_action(self, action: Dict[str, Any]):
        if not self.is_connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        left = np.asarray(action["left_joint_pos"], dtype=np.float32)
        right = np.asarray(action["right_joint_pos"], dtype=np.float32)

        self.left_arm.command_joint_pos(left)
        self.right_arm.command_joint_pos(right)

        return action

    def calibrate(self):
        print("Calibration not implemented. Use I2RT calibration tools.")