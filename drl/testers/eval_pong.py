"""
Evaluation script for Task 3: Enhanced DQN on Pong-v5
Compatible with task3.py where normalization (/255) happens OUTSIDE the model.
Usage: python src/eval_pong.py --model-path results/best_model.pt
"""
import argparse
import os
import random
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

gym.register_envs(ale_py)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)


class DQN(nn.Module):
    """Mirrors task3.py architecture — normalization is done externally."""

    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x)


def plot_evaluation_results(scores, seeds, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bars = axes[0].bar(range(len(scores)), scores, alpha=0.7,
                       edgecolor='black', color='green')
    axes[0].axhline(y=np.mean(scores), color='r', linestyle='--',
                    label=f'Mean: {np.mean(scores):.2f}')
    axes[0].axhline(y=19, color='b', linestyle=':', label='Target: 19')
    axes[0].set_xlabel('Episode (Seed)')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Task 3: Pong-v5 Evaluation Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(len(scores)))
    axes[0].set_xticklabels([f"S{s}" for s in seeds], rotation=45)
    for i, bar in enumerate(bars):
        if scores[i] >= 19:
            bar.set_color('darkgreen')

    axes[1].hist(scores, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=np.mean(scores), color='r', linestyle='--',
                    label=f'Mean: {np.mean(scores):.2f}')
    axes[1].axvline(x=19, color='b', linestyle=':', label='Target: 19')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cumulative = np.cumsum(scores)
    axes[2].plot(range(1, len(scores) + 1), cumulative, 'g-', marker='o', markersize=6)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Cumulative Score')
    axes[2].set_title('Cumulative Score Across Episodes')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Task 3 Evaluation Results (Seeds {seeds[0]}-{seeds[-1]})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.episodes))

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    # Strip `_orig_mod.` prefix added by torch.compile
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_frames:
        os.makedirs(os.path.join(args.output_dir, "frames"), exist_ok=True)

    all_scores = []
    episode_details = []

    print("=" * 60)
    print(f"Evaluating Model: {args.model_path}")
    print(f"Episodes: {args.episodes} | Device: {device}")
    print("=" * 60)

    for ep_idx, ep_seed in enumerate(seeds):
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        torch.manual_seed(ep_seed)

        obs, _ = env.reset(seed=ep_seed)
        state = preprocessor.reset(obs)

        done = False
        total_reward = 0
        frames = []
        step_count = 0

        while not done:
            if args.save_videos:
                frames.append(env.render())

            state_tensor = (
                torch.from_numpy(state)
                .unsqueeze(0)
                .to(device, dtype=torch.float32)
                .mul_(1.0 / 255.0)
            )
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if not done:
                state = preprocessor.step(next_obs)
            step_count += 1

        all_scores.append(total_reward)
        episode_details.append({
            'seed': ep_seed, 'score': total_reward, 'steps': step_count
        })
        print(f"  Ep {ep_idx+1:>2}/{args.episodes} | Seed {ep_seed:>2} | Score: {total_reward:>3.0f} | Steps: {step_count}")

        if args.save_videos and frames:
            vpath = os.path.join(args.output_dir, f"ep{ep_idx+1}_seed{ep_seed}_score{int(total_reward)}.mp4")
            with imageio.get_writer(vpath, fps=30) as video:
                for f in frames:
                    video.append_data(f)

        if args.save_frames and frames:
            for idx in [0, len(frames)//2, -1]:
                if abs(idx) < len(frames):
                    fpath = os.path.join(args.output_dir, "frames",
                                         f"ep{ep_idx+1}_seed{ep_seed}_frame{idx}.png")
                    imageio.imwrite(fpath, frames[idx])

    env.close()

    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    reached_target = sum(1 for s in all_scores if s >= 19)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Mean Score : {mean_score:.2f} ± {std_score:.2f}")
    print(f"  Min / Max  : {np.min(all_scores):.0f} / {np.max(all_scores):.0f}")
    print(f"  >= 19      : {reached_target}/{args.episodes} ({reached_target/args.episodes*100:.1f}%)")
    print("=" * 60)

    results_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Task 3 Evaluation — {args.model_path}\n")
        f.write("=" * 60 + "\n\n")
        for d in episode_details:
            f.write(f"Seed={d['seed']:>2}  Score={d['score']:>3.0f}  Steps={d['steps']}\n")
        f.write(f"\nMean: {mean_score:.2f} ± {std_score:.2f}\n")
        f.write(f">=19: {reached_target}/{args.episodes}\n")
    print(f"Results saved to {results_path}")

    plot_path = os.path.join(args.output_dir, "evaluation_plot.png")
    plot_evaluation_results(all_scores, seeds, plot_path)

    print(f"\nMean reward of {args.episodes} evaluation episodes: {mean_score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Task 3 DQN on Pong-v5")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Directory to save results")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (seeds 0..N-1)")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save .mp4 videos for each episode")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save keyframes as screenshots")

    args = parser.parse_args()
    evaluate(args)
