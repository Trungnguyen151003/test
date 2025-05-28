import gym
import numpy as np
import time
from td3_model import TD3
import spotdog_env  # để đăng ký env SpotDog-v0

# === Tạo môi trường với GUI ===
env = gym.make("SpotDog-v0", render=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# === Load mô hình đã huấn luyện ===
policy = TD3(state_dim, action_dim, max_action)
policy.load("TD3_SpotDog-v0_0", directory="./pytorch_models")

# === Reset môi trường ===
obs = env.reset()
done = False
episode_reward = 0

print("Replay simulation spotdog ")

while not done:
    # Lấy hành động từ chính sách đã học
    action = policy.select_action(np.array(obs))

    # Bước mô phỏng
    obs, reward, done, _ = env.step(action)
    episode_reward += reward

    # Tạm dừng một chút để PyBullet kịp hiển thị
    time.sleep(1.0 / 60.0)

print(f"✅ Total average reward episode: {episode_reward:.2f}")
env.close()
