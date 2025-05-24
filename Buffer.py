import random
import collections

import torch # 파이토치
import numpy as np # 넘파이

class Buffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 이전 조건들 저장용 deque

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) # 스탭마다 생기는 정보들 deque에 삽입

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 미니배치를 활용한 SGD를 위해 랜덤으로 배치사이즈만큼 미니배치 생성
        states, actions, rewards, next_states, dones = zip(*batch) # 배치 튜플 풀어서 각 변수 초기화

        states_np      = np.array(states, dtype=np.float32)      # shape: (B, obs_dim)
        next_states_np = np.array(next_states, dtype=np.float32) # shape: (B, obs_dim)
        actions_np     = np.array(actions, dtype=np.int64)       # shape: (B,)
        rewards_np     = np.array(rewards, dtype=np.float32)     # shape: (B,)
        dones_np       = np.array(dones, dtype=np.float32)       # shape: (B,)

        return (
            torch.from_numpy(states_np),      # FloatTensor (B, obs_dim)
            torch.from_numpy(actions_np),     # LongTensor  (B,)
            torch.from_numpy(rewards_np),     # FloatTensor (B,)
            torch.from_numpy(next_states_np), # FloatTensor (B, obs_dim)
            torch.from_numpy(dones_np),       # FloatTensor (B,)
        ) # torch의 tensor로 바로 바꾸면 VSC에서 속도 저하 경고가 뜨기 때문에 numpy에서 처리후 가져오기

    def __len__(self):
        return len(self.buffer) # 버퍼 deque 길이 반환