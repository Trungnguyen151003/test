import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

#Tạo một lớp để lưu buffer – nơi chứa các transitions (gồm: state, next_state, action, reward, done).
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = [] #nơi lưu các transition (s, s', a, r, done)
    self.max_size = max_size #tối đa bao nhiêu mẫu được lưu (mặc định: 1 triệu)
    self.ptr = 0 #chỉ số để vòng lặp ghi đè (overwrite) nếu bộ nhớ đã đầy

#Input: một transition, ví dụ: (state, next_state, action, reward, done)
# Nếu buffer chưa đầy, thêm vào cuối danh sách (append)
# Nếu buffer đã đầy, ghi đè vào vị trí ptr, rồi tăng ptr theo vòng lặp
  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size) #Lấy ngẫu nhiên batch_size số lượng chỉ số (index) trong buffer để huấn luyện.
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
# Tách từng phần từ các transition để phục vụ huấn luyện: state, next_state, action, reward, done
# Dùng np.array(..., copy=False) để tiết kiệm bộ nhớ
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.asarray(reward))
      batch_dones.append(np.asarray(done))

# Kết quả là 5 mảng NumPy tương ứng: states, next_states, actions, rewards, dones
# .reshape(-1, 1) để biến nó thành vector cột, chuẩn cho loss functions của PyTorch
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)