from i2rt.robots.get_robot import get_yam_robot

left = get_yam_robot(channel="can0")
obs = left.get_observations()

print(obs.keys())
print(obs["joint_pos"])
print(len