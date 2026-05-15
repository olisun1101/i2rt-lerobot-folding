from dataclasses import dataclass
import numpy as np

from i2rt.robots.get_robot import get_yam_robot


@dataclass
class I2RTBimanualTeleopConfig:
    left_leader_channel: str = "can_leader_l"
    right_leader_channel: str = "can_leader_r"

    left_gripper_type: str = "yam_teaching_handle"
    right_gripper_type: str = "yam_teaching_handle"

    action_dim_per_arm: int = 7


class I2RTBimanualTeleopAdapter:
    """
    把 I2RT leader arms 轉成 LeRobot action dict。
    """

    def __init__(self, config: I2RTBimanualTeleopConfig):
        self.config = config
        self.left_leader = None
        self.right_leader = None

    def connect(self):
        self.left_leader = get_yam_robot(
            channel=self.config.left_leader_channel,
            gripper_type=self.config.left_gripper_type,
        )

        self.right_leader = get_yam_robot(
            channel=self.config.right_leader_channel,
            gripper_type=self.config.right_gripper_type,
        )

        print("I2RT bimanual leader teleop connected.")

    def disconnect(self):
        self.left_leader = None
        self.right_leader = None
        print("I2RT bimanual leader teleop disconnected.")

    @property
    def is_connected(self):
        return self.left_leader is not None and self.right_leader is not None

    def get_action(self):
        """
        讀 leader arms，轉成 follower action。
        """

        obs_l = self.left_leader.get_observations()
        obs_r = self.right_leader.get_observations()

        # 先印一次確認 keys
        # print(obs_l.keys())

        left_q = np.asarray(obs_l["joint_pos"], dtype=np.float32)
        right_q = np.asarray(obs_r["joint_pos"], dtype=np.float32)

        # 如果 leader 是 6 維，但 follower action 是 7 維，這裡要補 gripper
        left_q = self._ensure_action_dim(left_q)
        right_q = self._ensure_action_dim(right_q)

        return {
            "left_joint_pos": left_q,
            "right_joint_pos": right_q,
        }

    def _ensure_action_dim(self, q):
        d = self.config.action_dim_per_arm

        if len(q) == d:
            return q

        if len(q) == 6 and d == 7:
            # 先給 gripper 一個預設值，之後你再改成 trigger input
            gripper = np.array([0.0], dtype=np.float32)
            return np.concatenate([q, gripper], axis=0)

        raise ValueError(f"Unexpected leader action dim: got {len(q)}, expected {d}")