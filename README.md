# DQN-for-CartPole-v1
Playing Atari with Deep Reinforcement Learning 논문을 바탕으로 cartpole v1 문제 해결

# 목차
  -문제
  
  -배경 설명
  
  -구현 방식
  
  -결과

## 문제
https://gymnasium.farama.org/environments/classic_control/cart_pole/ 에 소개되는  CartPole-v1 환경에 대해  https://arxiv.org/abs/1312.5602 논문에 소개된  DQN 강화학습 알고리즘으로 학습을 하여
Episode 전체 기간 동안 받은 Reward(즉, Episode Reward)의 총 합 500을 얻을 수 있는 에이전트를 개발하시오.
- 최종 결과물은 github 내에 웹 페이지를 구성하여 Markdown으로 작성하여 제시하시오.
- 해당 github 페이지 내에 프로그램 소스 코드에 대한 링크를 걸어서 직접 소스 코드를 검토할 수 있도록 하시오 (코드를 여러 코드로 나누었다면 나누어진 각 코드별로 링크 제시).
- 소스 코드 내에는 주석을 한글로 또는 영어로 충분하게 제시하시오.
- 해당 github 페이지 내에 Episode Reward로서 500을 얻어 낸 화면을 캡쳐하여 해당 이미지를 함께 보이도록 하시오.
- 해당 github 페이지 내에 위 에이전트가  Cartpole을 오랫동안 세우는 영상을 임베드하고 클릭을 하면 영상이 플레이되도록 하시오.

## 배경 설명

### 1. 행동 집합
$$
A = \{1, \ldots, K\}
$$

Cartpole에서 행동 집합은 앞과 뒤 2가지이다.

### 2. 관측 및 상태 정의
$$
x_t \in \mathbb{R}^d, \quad r_t \quad\text{(Reward)}
s_t = x_1, a_1, x_2, \ldots, a_{t-1}, x_t
$$

카트폴의 상태를 관측하고 상태는 4차원 벡터 형태로 나타난다.

카트 위치는 -4.8 ~ 4.8 사이의 값을 가진다.

카트 속도는 -Inf ~ Inf 사이의 값을 가진다.

폴의 각도는 -24도 ~ 24도 사이의 값을 가진다.

폴의 각속도는 -Inf ~ Inf 사이의 값을 가진다.

### 3. 할인된 누적 보상
$$
R_t = \sum_{t'=t}^{T} \gamma^{\,t'-t}\,r_{t'}
$$

한 에피소드 내에서 보상을 할인율을 적용하여 미래 보상의 합을 계산한다.
사용한 $$\gamma$$ 는 0.99를 사용하였다.

### 4. 최적 행동 가치 함수 정의

$$
Q^*(s,a) = \max_{\pi}\,\mathbb{E}\bigl[R_t \mid s_t = s,\,a_t = a,\,\pi\bigr]
$$

Q는 상태 𝑠에서 행동 𝑎를 취하고, 이후 최적인 정책 π를 따라갔을 때 기대할 수 있는 장기 누적 보상의 최대값이다.

### 5. 벨만 최적 방정식

$$
Q^*(s,a) = E\bigl[\,r + \gamma\,\max_{a'}\,Q^\star(s',a') \mid s,a\,\bigr]
$$

벨만 최적 방정식은 Q란 상태 s에 대해 행동 a들에 대해서 현재 보상과 다음 상태 s'에서 취하는 행동 a'가 주는 미래보상을 합한 것을 최대화 하는 행동 a를 선택하는 것이다.

단순히 매 스탭마다 현재 최대의 보상만 보는 것이 아니라 할인된 다음의 미래보상까지 고려하여 보상을 최대화하는 재귀적인 방법이다.

### 6. 값 반복 업데이트

$$
Q_{i+1}(s,a) = \mathbb{E}\bigl[\,r + \gamma \max_{a'}Q_i(s',a') \mid s,a\bigr]
$$

이전의 Q를 이용하여 상태 s'와 행동 a'를 업데이트한다.

### 7. 손실 함수 정의

$$
L_i(\theta_i)= 
E_{s,a \sim \rho}\Bigl[\;\bigl(y_i - Q(s,a;\theta_i)\bigr)^2\Bigr]
$$

### 8. 손실 함수의 그래디언트

$$
\nabla_{\theta_i} L_i(\theta_i)= 
E_{s,a\sim\rho,\,s'}\Bigl[\,
   \Bigl(r + \gamma\,\max_{a'}Q(s',a';\theta_{i-1}) 
       - Q(s,a;\theta_i)\Bigr)\,
   \nabla_{\theta_i} Q(s,a;\theta_i)
\Bigr]
$$

여기서 핵심은 이전 파라미터를 가진 QNetwork가 필요하다는 것이다.

### 9. 업데이트 시 사용되는 타깃 값

$$
y_j =
\begin{cases}
  r_j,
  &\text{if }\phi_{j+1}\text{ is terminal},\\
  r_j + \gamma \max_{a'}Q(\phi_{j+1},a';\theta),
  &\text{otherwise}.
\end{cases}
$$

## 구현 방식

확률이 포함된 탐욕적 방법으로 상태에 따른 행동 조합을 탐색한다.
```python
    #E-greedy 기법
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * max(0, (epsilon_decay - step) / epsilon_decay)
    state_t = torch.tensor(state, dtype=torch.float32)
    if random.random() < epsilon:
        action = random.randrange(n_actions) # 아무 행동이나 선택(데이터 다양성을 늘려 초기 학습에 도움)
    else:
        with torch.no_grad(): # 자동 그라디언트 계산이 순전파만 할 시에는 필요없으므로 미분트리를 생성하지 않는다.
            action = policy_net(state_t.unsqueeze(0)).argmax(1).item() # 네트워크를 통해 얻은 가치에서 최고 가치를 지니는 행동 선택(탐욕적)
```

샘플링된 상태 환경 조합을 사용하여 miniBatch학습을 진행한다.
Batch 학습이기 때문에 배치의 기울기를 동시에 고려해서 학습하고
과거 Replay를 활용하는 논문의 기작과도 일치한다.
```python
        # 이전 파라미터를 타깃용으로 복사
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
```

## 결과



![image](https://github.com/user-attachments/assets/b9fb65be-31aa-4a7e-a631-48bda8ef2ca1)

https://github.com/user-attachments/assets/3eb47547-e380-42a3-8290-9ca912b56ee9

