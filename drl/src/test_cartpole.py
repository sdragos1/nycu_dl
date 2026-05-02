import argparse
import random

import gymnasium as gym
import numpy as np
import torch

from src.task1 import DQN


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    num_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]

    model = DQN(input_dim, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    ep_rewards = []
    env_cnt = 0
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env_cnt += 1

        ep_rewards.append(total_reward)

    avg_ep_reward = sum(ep_rewards) / len(ep_rewards)
    print(f"Environment steps: {env_cnt}, seed: {args.seed}, average reward: {avg_ep_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
