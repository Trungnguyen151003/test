import gym
import numpy as np
import torch
from td3_model import TD3
from replay_buffer import ReplayBuffer
from utilss import evaluate_policy, mkdir
import spotdog_env
# === Cấu hình huấn luyện ===
env_name = "SpotDog-v0"
start_timesteps = 1e4       # Random actions trước khi huấn luyện
eval_freq = 5e3             # Bao nhiêu bước thì evaluate 1 lần
max_timesteps = 2e6         # Tổng số bước
expl_noise = 0.1            # Noise cho hành động
batch_size = 100
discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
seed = 0

# === Khởi tạo môi trường ===
env = gym.make(env_name, render=True )
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# === Khởi tạo mô hình TD3 và Replay Buffer ===
policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()
evaluations = []

# === Tạo thư mục lưu kết quả ===
mkdir(".", "results")
mkdir(".", "pytorch_models")

file_name = f"TD3_{env_name}_{seed}"
print("---------------------------------------")
print(f"Training environment: {env_name}")
print("---------------------------------------")

# === Biến huấn luyện ===
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True

# === Vòng huấn luyện chính ===
while total_timesteps < max_timesteps:
    if done:
        if total_timesteps != 0:
            print(f"Total Timesteps: {total_timesteps} Episode Num: {episode_num} Reward: {episode_reward:.2f}")
            policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau,
                         policy_noise, noise_clip, policy_freq)

        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy, env))
            policy.save(file_name, directory="./pytorch_models")
            np.save(f"./results/{file_name}", evaluations)

        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if total_timesteps < start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.select_action(np.array(obs))
        action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
            env.action_space.low, env.action_space.high)

    new_obs, reward, done, _ = env.step(action)

    done_bool = 0 if done else 1
    replay_buffer.add((obs, new_obs, action, reward, done_bool))

    obs = new_obs
    episode_reward += reward
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# === Đánh giá và lưu lần cuối ===
evaluations.append(evaluate_policy(policy, env))
policy.save(file_name, directory="./pytorch_models")
np.save(f"./results/{file_name}", evaluations)
