import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class QNetworkCartPole(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(QNetworkCartPole, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


def evaluate_model(model_path, num_episodes=20, seed_start=0, render=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    model = QNetworkCartPole().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()

    rewards = []
    seeds = list(range(seed_start, seed_start + num_episodes))

    print(f"Evaluating model: {model_path}")
    print(f"Seeds: {seeds}")
    print("-" * 50)

    for episode_idx, seed in enumerate(seeds):
        env.reset(seed=seed)
        state, _ = env.reset(seed=seed)

        episode_reward = 0
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step_count += 1

        rewards.append(episode_reward)
        print(
            f"Episode {episode_idx + 1:2d} | Seed: {seed:3d} | Reward: {episode_reward:5.1f} | Steps: {step_count:4d}")

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print("-" * 50)
    print(f"Evaluation Results over {num_episodes} episodes:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")

    score_percentage = min(mean_reward, 480) / 480 * 15
    print(f"Score Percentage: {score_percentage:.2f}% (out of 15%)")

    return rewards, seeds, mean_reward, std_reward


def plot_results(rewards, seeds, save_path="eval_task1_results.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(rewards)), rewards, alpha=0.7, edgecolor='black')
    ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    ax1.fill_between(range(len(rewards)),
                     np.mean(rewards) - np.std(rewards),
                     np.mean(rewards) + np.std(rewards),
                     alpha=0.2, color='r')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Task 1: CartPole-v1 Evaluation Results')
    ax1.set_ylim([0, 550])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.set_xticks(range(len(rewards)))
    ax1.set_xticklabels([f"S{seed}" for seed in seeds], rotation=45)

    ax2.plot(range(1, len(rewards) + 1), np.cumsum(rewards), 'b-', marker='o', markersize=4)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards Across Episodes')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN on CartPole-v1")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Starting seed for evaluation")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--save_plot", type=str, default="eval_task1_results.png",
                        help="Path to save the plot")

    args = parser.parse_args()

    rewards, seeds, mean_reward, std_reward = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        render=args.render
    )

    plot_results(rewards, seeds, save_path=args.save_plot)

    print("\n" + "=" * 50)
    print("SUMMARY FOR REPORT:")
    print(f"Model: {args.model_path}")
    print(f"Mean Reward over 20 episodes: {mean_reward:.2f}")
    print(f"Standard Deviation: {std_reward:.2f}")
    print("=" * 50)
