# i2rt-lerobot-folding

本專案用於雙臂機器人摺疊任務的數據採集與控制，結合 i2rt 與 LeRobot 框架。

## 🛠️ 環境架設

```bash
# 建立 conda 環境
conda create -n "folding" python=3.12 -y
conda activate folding

# 安裝依賴項目
# 請參考：[https://github.com/i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt)
# 請參考：[https://huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation)


can_leader_l  ──┐
                ├─ I2RTBimanualTeleopAdapter.get_action()
can_leader_r  ──┘
                        ↓
                  action dict
                        ↓
can_follower_l ──┐
                 ├─ I2RTDualArmRobot.send_action()
can_follower_r ──┘
                        ↓
         I2RTDualArmRobot.get_observation()
         ├─ follower joint state
         ├─ top camera
         ├─ left camera
         └─ right camera
                        ↓
                 LeRobot dataset
