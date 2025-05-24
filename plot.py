import numpy as np
import matplotlib.pyplot as plt # 그래프 시각화용

def print_plot(predict_maxQ, episode_rewards, losses):
    # 학습 이후 한 번에 그리기
    window = 100 # 스무딩을 위한 창문
    plt.figure(figsize=(12,6))

    # 최대 Q값 출력
    plt.subplot(1, 3, 1)
    plt.plot(predict_maxQ)
    plt.xlabel('Steps')
    plt.ylabel('Average Max Q')
    plt.title('Max Q')

    # 에피소드 리턴 추이 (이동 평균)
    plt.subplot(1, 3, 2)
    smoothed_episode_rewards = [np.mean(episode_rewards[i-window:i+1]) if i>=window else losses[i] for i in range(len(episode_rewards))]
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward')

    # 손실 추이 (이동 평균)
    smoothed_loss = [np.mean(losses[i-window:i+1]) if i>=window else losses[i] for i in range(len(losses))]
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Smoothed Loss')
    plt.title('Training Loss')

    plt.show()