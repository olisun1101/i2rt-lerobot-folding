from lerobot.robots import make_robot

robot = make_robot("i2rt_dual_arm")

robot.connect()

obs = robot.get_observation()
print("OBS:", obs)

# 隨便動一下
action = {
    "left_joint_pos": [0, 0, 0, 0, 0, 0],
    "right_joint_pos": [0, 0, 0, 0, 0, 0],
}

robot.send_action(action)
``