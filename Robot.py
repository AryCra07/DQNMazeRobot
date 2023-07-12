import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from QRobot import QRobot

import torch
import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Robot(QRobot):
    def __init__(self, maze):
        super(Robot, self).__init__(maze)
        # 添加自定义参数和初始化逻辑
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []  # 用于存储训练数据的经验回放缓冲区

    def build_model(self):
        input_dim = 2  # 根据实际情况设置输入维度
        output_dim = 4  # 根据实际情况设置输出维度
        model = DQNModel(input_dim, output_dim)
        return model


    def train_update(self):
        state = self.sense_state()  # 获取当前状态
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.model(state)  # 通过模型获取当前状态下各个动作的Q值

        # 根据Q值选择动作
        action = torch.argmax(q_values).item()

        reward = self.maze.move_robot(action)  # 执行动作并获取奖励值

        next_state = self.sense_state()  # 获取下一个状态
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_q_values = self.target_model(next_state)  # 通过目标模型获取下一个状态下各个动作的Q值

        # 计算目标Q值
        max_next_q_value = torch.max(next_q_values).item()
        target_q_value = reward + self.discount_factor * max_next_q_value

        # 计算当前Q值
        current_q_value = q_values[0][action]

        # 计算损失函数
        loss = self.criterion(current_q_value, target_q_value)

        # 更新模型参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return action, reward


def test_update(self):
    state = self.sense_state()  # 获取当前状态
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    q_values = self.model(state)  # 通过模型获取当前状态下各个动作的Q值

    # 根据Q值选择动作
    action = torch.argmax(q_values).item()

    reward = self.maze.move_robot(action)  # 执行动作并获取奖励值

    return action, reward