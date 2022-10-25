import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from icecream import ic

class Conv_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, r):
        super().__init__()
        assert type(r) is int
        assert r >=4, f'You chose r={r}. Please choose r>3.'
        self.r = r
        self.linear1 = nn.Linear(input_size, hidden_size)
        # matrix_size = 2*r+1
        self.conv1 = nn.Conv2d(2, 4, 4) # N, 4, 2*r-2, 2*r-2 
        self.pool1 = nn.MaxPool2d(2, 2) # N, 4, r-1, r-1
        self.conv2 = nn.Conv2d(4, 8, 3) # N, 8, r-3, r-3
        
        self.fc1 = nn.Linear(int(8 * (r-3)**2), hidden_size)
        self.linear2 = nn.Linear(2*hidden_size, output_size)
        
    def forward(self, state1, state2):
        x1 = F.relu(self.linear1(state1))
        x2 = self.conv1(state2)
        x2 = self.pool1(F.relu(x2))
        x2 = F.relu(self.conv2(x2))
        
        x1 = torch.reshape(x1,(x1.shape[0],-1))
        x2 = torch.reshape(x2,(x1.shape[0],-1))
        
        x2 = self.fc1(x2)
        x = torch.concat((x1,x2), axis = 1)
        x = self.linear2(x)
        return x

    def save(self, file_name='test_conv.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, state1, state2):
        x1 = F.relu(self.linear1(state1))
        x = self.linear2(x1)
        return x

    def save(self, file_name='test_lin.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state1, state2, action, reward, next_state1,next_state2, done):
        state1 = torch.tensor(state1, dtype=torch.float)
        state2 = torch.tensor(np.array(state2).copy(), dtype=torch.float)
        next_state1 = torch.tensor(next_state1, dtype=torch.float)
        next_state2 = torch.tensor(np.array(next_state2).copy(), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state1.shape) == 1:
            # (1, x)
            state1 = torch.unsqueeze(state1, 0)
            state2 = torch.unsqueeze(state2, 0)
            next_state1 = torch.unsqueeze(next_state1, 0)
            next_state2 = torch.unsqueeze(next_state2, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        # 1: predicted Q values with current state
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        pred = self.model(state1, state2)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state1[idx].unsqueeze(0), next_state2[idx].unsqueeze(0)))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        




