import time
import numpy as np

from your_robot_wrapper import I2RTDualArmRobot, I2RTDualArmConfig
from i2rt_lerobot_folding.teleop_adapter import (
    I2RTBimanualTeleopAdapter,
    I2RTBimanualTeleopConfig,
)


def main():
    robot_config = I2RTDualArmConfig(
        left_channel="can_follower_l",
        right_channel="can_follower_r",

        # 如果你現在先不用 RealSense，可換回你的 camera config
        # realsense_top_serial="PUT_TOP_SERIAL",
        # realsense_left_serial="PUT_LEFT_SERIAL",
        # realsense_right_serial="PUT_RIGHT_SERIAL",

        action_dim_per_arm=7,
    )

    teleop_config = I2RTBimanualTeleopConfig(
        left_leader_channel="can_leader_l",
        right_leader_channel="can_leader_r",
        action_dim_per_arm=7,
    )

    robot = I2RTDualArmRobot(robot_config)
    teleop = I2RTBimanualTeleopAdapter(teleop_config)

    robot.connect()
    teleop.connect()

    fps = 30
    dt = 1.0 / fps

    trajectory = []

    print("Move leader/follower to similar start pose before recording.")
    input("Press Enter to start recording...")

    try:
        while True:
            t0 = time.time()

            # 1. 從 leader arms 取得 action
            action = teleop.get_action()

            # 2. 送給 follower arms
            robot.send_action(action)

            # 3. 讀 follower state + cameras
            obs = robot.get_observation()

            # 4. 存一筆
            step = {
                "timestamp": time.time(),
                "observation": obs,
                "action": action,
            }
            trajectory.append(step)

            elapsed = time.time() - t0
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stop recording.")

    finally:
        teleop.disconnect()
        robot.disconnect()

    print(f"Recorded {len(trajectory)} steps.")

    # 先簡單存 npz 測試，之後再轉 LeRobotDataset
    np.save("debug_trajectory.npy", trajectory, allow_pickle=True)
    print("Saved debug_trajectory.npy")


if __name__ == "__main__":
    main()