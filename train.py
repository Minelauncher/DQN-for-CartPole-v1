import random

import gymnasium as gym # 카트폴 환경 구현용
import torch # 파이토치
import torch.optim as optim # 파이토치 옵티마이저 부분(ADAM 옵티마이저 사용을 위해서)

from QNetwork import QNetwork
from Buffer import Buffer
import plot

# 파일 실행 방법
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#. .\.venv\Scripts\Activate.ps1
#python 파일이름.py
    
# CartPole 환경 생성
env = gym.make("CartPole-v1") # 카트폴 환경 생성 v1은 스탭당 보상이 1씩 누적되고 500이 한도
obs_dim = env.observation_space.shape[0] # 카트폴 상태 벡터 길이: 4
n_actions = env.action_space.n # 좌우 행동 길이: 2

policy_net = QNetwork(obs_dim, 128, n_actions) # 타겟 네트워크를 따로 두지 않고 네트워크 인스턴스 하나만 생성
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3) # 무난한 ADAM 옵티마이저. 학습률을 조절하여 수렴에 도움을 준다.
replay_buffer = Buffer(10000) # 리플레이 버퍼 인스턴스 생성

gamma = 0.99 # 미래 보상에 대한 할인율
batch_size = 32 # 미니배치 사이즈

# 가치 높으면 바로 선택하는 탐욕적 기법을 기반으로 하지만 학습 초반에는 행동의 가치가 보장되지 않으므로 확률적으로 다른 행동을 취해 데이터 다양성을 높인다.
epsilon_start = 1.0 
epsilon_final = 0.1
epsilon_decay = 20000 

num_steps = 100000 # 환경과 상호작용할 단위시간의 수(step의 수)

# 시각화를 위한 기록용 리스트 준비
episode_rewards = [] # 에피소드 당 보상 리스트(이때, 에피소드는 실패 혹은 성공시 초기화되면서 시작)
losses = [] # 손실 리스트
predict_maxQ = [] # 예측 최대 Q 리스트

best_saved = False # 목표에 도달한 저장본이 있는가?

state, _ = env.reset(seed=0) # 상태 초기화(시드를 고정한다) (-0.05, 0.05)도 내에서 초기화
episode_reward = 0 # 에피소드 당 보상 변수 초기화


# Q값 추정용 초기 조건 모으기
buffer = Buffer(100)
for i in range(100):
    action = random.randrange(n_actions)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated # 끝났는가?
    buffer.push(state, action, reward, next_state, done) # 행동 전후로 데이터 셋 버퍼에 삽입
    if done:
        state, _ = env.reset() # 상태 초기화


# 학습 루프
state, _ = env.reset(seed=0) # 상태 초기화(시드를 고정한다) (-0.05, 0.05)도 내에서 초기화
for step in range(1, num_steps + 1):

    # E-greedy 기법
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * max(0, (epsilon_decay - step) / epsilon_decay)
    state_t = torch.tensor(state, dtype=torch.float32)
    if random.random() < epsilon:
        action = random.randrange(n_actions) # 아무 행동이나 선택(데이터 다양성을 늘려 초기 학습에 도움)
    else:
        with torch.no_grad(): # 자동 그라디언트 계산이 순전파만 할 시에는 필요없으므로 미분트리를 생성하지 않는다.
            action = policy_net(state_t.unsqueeze(0)).argmax(1).item() # 네트워크를 통해 얻은 가치에서 최고 가치를 지니는 행동 선택(탐욕적)


    next_state, reward, terminated, truncated, _ = env.step(action) # 행동이 취해진 후 상태(환경)가 변화함
    done = terminated or truncated # 끝났는가?
    replay_buffer.push(state, action, reward, next_state, done) # 행동 전후로 데이터 셋 리플레이 버퍼에 삽입
    state = next_state # 상태 갱신(환경)


    episode_reward += reward # 스탭 당 보상 누적


    # 버퍼 충분할 때만 업데이트(미니배치 수를 넘는 시점부터)
    if len(replay_buffer) >= batch_size:
        # 이전 파라미터를 타깃용으로 복사 (freeze)
        target_net = QNetwork(obs_dim, 128, n_actions)
        target_net.load_state_dict(policy_net.state_dict())

        # 배치 샘플링
        S, A, R, S_next, D = replay_buffer.sample(batch_size)

        # 현재 Q값
        q_values = policy_net(S).gather(1, A.unsqueeze(1)).squeeze(1)

        # 타깃 계산 (이전 파라미터로만)
        with torch.no_grad():
            q_values_next = target_net(S_next) # shape: (batch_size, action_dim)
            max_vals, max_idxs = q_values_next.max(dim=1) # 행동 차원 기준 최고값 선택 0,1인데 행동은 1에
            q_next = max_vals # max_vals는 탐욕적 방법으로 가치 높은 것 선택 shape: (batch_size,)
            target = R + gamma * q_next * (1-D) # D가 참이면 끝이므로 미래 보상 고려할 필요가 없다.

        # 손실 및 역전파
        loss = torch.nn.functional.mse_loss(q_values, target) # 토치 내부에 구현되어 있는 MSE 손실함수 사용
        optimizer.zero_grad() # 옵티마이저 초기화
        loss.backward() # 자동 그라디언트 전파
        optimizer.step() # 옵티마이저 적용

        losses.append(loss.item()) # 여기에 loss 저장(item으로 tensor를 숫자로 변환)


        with torch.no_grad(): # 자동 그라디언트 계산이 순전파만 할 시에는 필요없으므로 미분트리를 생성하지 않는다.
            S, A, R, S_next, D = buffer.sample(100)
            q_values = policy_net(S_next)
            max_q_values, _ = q_values.max(dim=1)
            avg_max_q = max_q_values.mean().item() # (item으로 tensor를 숫자로 변환)
            predict_maxQ.append(avg_max_q) # Q


    if done: # 에피소드 끝나면 리셋
        episode_rewards.append(episode_reward) # 에피소드 보상 저장
        
        if not best_saved and episode_reward >= 500: # 에피소드가 보상 500을 달성하여 성공하였다면
            torch.save(policy_net.state_dict(), "checkpoints/best_policy.pth") # 상대경로에 네트워크 저장
            print("보상",episode_reward," 달성. 모델 저장 후 학습 종료.")
            best_saved = True # 가장 좋은 저장본 존재
            break # 학습의 목적을 달성했으므로 학습 탈출

        episode_reward = 0 # 에피소드 보상 초기화
        state, _ = env.reset() # 상태 초기화
        
env.close() # 카트폴 환경 종료


# 저장된 모델로 렌더링
if best_saved:
    print("▶ 저장된 정책으로 재생 시작")
    # 모델 로드
    policy_net.load_state_dict(torch.load("checkpoints/best_policy.pth"))

    # 렌더 모드 human 으로 새 환경
    render_env = gym.make("CartPole-v1", render_mode="human")
    state, _ = render_env.reset(seed=0)
    done = False
    while not done:
        with torch.no_grad():
            action = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(1).item()
        state, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        render_env.render()
    render_env.close()


plot.print_plot(predict_maxQ=predict_maxQ, episode_rewards=episode_rewards, losses=losses)