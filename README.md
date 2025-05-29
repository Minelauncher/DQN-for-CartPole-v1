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
x_t \in \mathbb{R}^d, \quad r_t \quad\text{(Reward)   }
s_t = x_1, a_1, x_2, \ldots, a_{t-1}, x_t
$$



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

### 5. Bellman 최적 방정식
$$
Q^*(s,a) = \mathbb{E}_{s'\sim E}\bigl[r + \gamma \max_{a'}Q^*(s',a') \mid s,a\bigr]
$$

### 6. 값 반복 업데이트
$$
Q_{i+1}(s,a) = \mathbb{E}\bigl[\,r + \gamma \max_{a'}Q_i(s',a') \mid s,a\bigr]
$$

### 7. 손실 함수 정의
$$
L_i(\theta_i) 
  = \mathbb{E}_{s,a\sim\rho(\cdot)}
    \Bigl[\bigl(y_i - Q(s,a;\theta_i)\bigr)^2\Bigr],
\quad
y_i = \mathbb{E}_{s'\sim E}\bigl[\,r + \gamma \max_{a'}Q(s',a';\theta_{i-1}) \mid s,a\bigr]
$$

### 8. 손실 함수의 그래디언트
$$
\nabla_{\theta_i}L_i(\theta_i)
  = \mathbb{E}_{s,a\sim\rho(\cdot);\,s'\sim E}
    \Bigl[\bigl(r + \gamma \max_{a'}Q(s',a';\theta_{i-1})
                   - Q(s,a;\theta_i)\bigr)\,
          \nabla_{\theta_i}Q(s,a;\theta_i)\Bigr]
$$

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

## 결과

https://github.com/user-attachments/assets/3eb47547-e380-42a3-8290-9ca912b56ee9

