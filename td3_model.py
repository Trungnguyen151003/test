import torch
import torch.nn as nn
import torch.nn.functional as F
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Actor(nn.Module):
    # state_dim: Số chiều của trạng thái đầu vào (ví dụ: 3 hoặc 8...)
    # action_dim: Số chiều của hành động đầu ra (ví dụ: 1 hoặc 4)
    # max_action: Giá trị tuyệt đối lớn nhất của hành động (dùng để scale đầu ra)
    def __init__(self, state_dim, action_dim, max_action):
        # Đây là mạng fully-connected gồm 3 lớp:
        # Input → 400
        # 400 → 300
        # 300 → action_dim
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action #Dùng để đảm bảo giá trị hành động không vượt quá giới hạn cho phép

    def forward(self, x):
        x = F.relu(self.layer_1(x)) # ReLU để tạo phi tuyến tính
        x = F.relu(self.layer_2(x)) # ReLU tiếp tục
        x = self.max_action * torch.tanh(self.layer_3(x)) # Scale đầu ra
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)  #Ghép trạng thái x và hành động u thành một vector đầu vào [state | action]
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    # Hàm này chỉ chạy Critic đầu tiên
    # Dùng trong quá trình cập nhật Actor, vì Actor không cần cả hai Q, chỉ dùng Q1 để tính loss:
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device) #actor: Mạng chọn hành động (hàm chính sách π(s))
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device) # actor_target: Bản sao dùng để cập nhật mượt (soft update), giúp huấn luyện ổn định hơn.
        self.actor_target.load_state_dict(self.actor.state_dict()) #load_state_dict: Copy toàn bộ trọng số từ actor sang actor_target.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters()) #Dùng optimizer Adam để cập nhật trọng số của actor.
        self.critic = Critic(state_dim, action_dim).to(device) #critic: Dự đoán Q(s, a) từ hai mạng Critic.
        self.critic_target = Critic(state_dim, action_dim).to(device) #critic_target: Bản sao của critic, dùng cho việc tính Q-target (mục tiêu huấn luyện).
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action #Dùng để scale lại hành động đầu ra từ tanh của Actor
        self.device = device

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device) # Nếu bạn đang làm việc với môi trường gym, môi trường sẽ trả state dưới dạng np.array,
                                                                   # phải chuyển từ NumPy ➜ Tensor
        return self.actor(state).cpu().data.numpy().flatten() #Đưa kết quả từ GPU về CPU → NumPy → flatten thành mảng 1D.

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min (Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2,
                                                                        target_Q)  # Mục tiêu là giảm sai lệch giữa Q hiện tại và Q-target.

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()  # Xóa các gradient cũ đang tích lũy từ bước training trước đó. Nếu không gọi dòng này, gradient sẽ cộng dồn, dẫn đến cập nhật sai lệch.
            critic_loss.backward()  # Tính gradient của hàm mất mát critic_loss đối với tất cả tham số của mạng Critic.Dựa trên autograd, PyTorch sẽ lan truyền ngược từ loss về từng layer để tính đạo hàm
            self.critic_optimizer.step()  # Dựa trên gradient vừa tính, cập nhật các trọng số của Critic.nó sẽ cập nhật theo công thức Adam

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
# Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

  # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))