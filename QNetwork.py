import torch                                     # 파이토치
import torch.nn as nn                            # 파이토치 신경망 부분
# Q 네트워크 (상태를 입력으로 행동가치를 출력으로 삼는 Full-Connected Layers)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super().__init__() # nn.Module 상속받는 클래스라 생성자 호출 추가적으로 기존 메서드이용 가능
        self.fc1 = nn.Linear(obs_dim, hidden_dim) # Layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # Layer
        self.fc3 = nn.Linear(hidden_dim, n_actions) # Layer

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # 비선형성 추가용 ReLU
        x = torch.relu(self.fc2(x)) # 비선형성 추가용 ReLU
        return self.fc3(x)  # Q-value 높으면 그 행동을 취함