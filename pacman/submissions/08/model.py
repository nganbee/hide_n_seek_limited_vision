# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PacmanNet(nn.Module):
    def __init__(self, input_shape=(1, 21, 21), n_actions=4):
        super(PacmanNet, self).__init__()
        
        # CNN để nhìn bản đồ (Map Feature Extraction)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Tính toán kích thước sau khi qua CNN (Map 21x21)
        self.feature_size = 32 * 21 * 21
        
        # Layer kết hợp: Map Features + Last Move (để học quán tính)
        # Input size = Feature map + 4 (One-hot vector của hướng cũ)
        self.fc1 = nn.Linear(self.feature_size + 4, 256)
        self.fc2 = nn.Linear(256, n_actions) # Output: Q-values cho 4 hướng

    def forward(self, x, last_move_vec):
        # x: [Batch, 1, 21, 21] - Bản đồ
        # last_move_vec: [Batch, 4] - Vector hướng cũ (VD: [0,1,0,0])
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Duỗi phẳng ảnh ra
        
        # Nối ảnh và hướng lại với nhau
        combined = torch.cat((x, last_move_vec), dim=1)
        
        x = F.relu(self.fc1(combined))
        return self.fc2(x)