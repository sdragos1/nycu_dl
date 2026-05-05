"""
Evaluation script for Task 2 & 3: DQN on Pong-v5
Follows professor's evaluation protocol with seeds 0-19
Usage: python eval_pong.py --model_path LAB5_StudentID_task2.pt --task 2
"""

import argparse
import os
import random
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.task2 import AtariPreprocessor


class DQN(nn.Module):
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
        return self.network(x / 255.0)


def plot_evaluation_results(scores, seeds, task, save_path):
    """Plot evaluation results for report"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Bar plot
    bars = axes[0].bar(range(len(scores)), scores, alpha=0.7,
                       edgecolor='black', color='green')
    axes[0].axhline(y=np.mean(scores), color='r', linestyle='--',
                    label=f'Mean: {np.mean(scores):.2f}')
    axes[0].axhline(y=19, color='b', linestyle=':', label='Target: 19')
    axes[0].set_xlabel('Episode (Seed)')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Task {task}: Pong-v5 Evaluation Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(len(scores)))
    axes[0].set_xticklabels([f"S{seed}" for seed in seeds], rotation=45)

    # Color bars that meet target
    for i, bar in enumerate(bars):
        if scores[i] >= 19:
            bar.set_color('darkgreen')
            bar.set_edgecolor('black')

    # Histogram
    axes[1].hist(scores, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=np.mean(scores), color='r', linestyle='--',
                    label=f'Mean: {np.mean(scores):.2f}')
    axes[1].axvline(x=19, color='b', linestyle=':', label='Target: 19')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Cumulative sum
    cumulative = np.cumsum(scores)
    axes[2].plot(range(1, len(scores) + 1), cumulative, 'g-', marker='o', markersize=6)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Cumulative Score')
    axes[2].set_title('Cumulative Score Across Episodes')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Task {task} Evaluation Results (Seeds {seeds[0]}-{seeds[-1]})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use seeds 0 to episodes-1 as per professor's instruction
    # "Evaluation seeds: 0 to 19"
    seeds = list(range(args.episodes))

    # Create environment
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    # Load model
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

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "frames"), exist_ok=True)

    # Store results
    all_scores = []
    all_rewards = []
    episode_details = []

    print("=" * 60)
    print(f"Evaluating Model: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Number of episodes: {args.episodes}")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    for ep_idx, seed in enumerate(seeds):
        print(f"\nEpisode {ep_idx + 1}/{args.episodes} (Seed: {seed})")

        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)

        done = False
        truncated = False
        total_reward = 0
        episode_score = 0  # Track Pong score (points scored - points conceded)
        frames = []
        step_count = 0

        while not (done or truncated):
            # Capture frame for video
            frame = env.render()
            frames.append(frame)

            # Get action from model (greedy, no exploration)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Track actual score (reward in Pong is typically +1 for scoring, -1 for conceding)
            if reward != 0:
                episode_score += reward

            # Update state
            if not (done or truncated):
                state = preprocessor.step(next_obs)

            step_count += 1

        # Store results
        all_scores.append(episode_score)
        all_rewards.append(total_reward)
        episode_details.append({
            'seed': seed,
            'score': episode_score,
            'total_reward': total_reward,
            'steps': step_count,
            'frames': len(frames)
        })

        print(f"  Score: {episode_score} | Total Reward: {total_reward:.1f} | Steps: {step_count}")

        # Save video for this episode
        if args.save_videos:
            video_path = os.path.join(args.output_dir, f"task{args.task}_ep{ep_idx+1}_seed{seed}_score{episode_score}.mp4")
            with imageio.get_writer(video_path, fps=30) as video:
                for f in frames:
                    video.append_data(f)
            print(f"  Video saved to: {video_path}")

        # Save a few keyframes as screenshots for report
        if args.save_frames and frames:
            # Save first, middle, and last frames
            key_indices = [0, len(frames)//2, -1]
            for idx in key_indices:
                if abs(idx) < len(frames):
                    frame_path = os.path.join(args.output_dir, "frames",
                                              f"task{args.task}_ep{ep_idx+1}_seed{seed}_frame{idx}.png")
                    imageio.imwrite(frame_path, frames[idx])

    env.close()

    # Calculate statistics
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)

    # Count episodes that reached target
    reached_target = sum(1 for s in all_scores if s >= 19)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes evaluated: {args.episodes}")
    print(f"Seeds used: {seeds[0]} to {seeds[-1]}")
    print(f"\nScore Statistics:")
    print(f"  Mean Score: {mean_score:.2f} ± {std_score:.2f}")
    print(f"  Min Score: {min_score}")
    print(f"  Max Score: {max_score}")
    print(f"  Episodes reaching score 19: {reached_target}/{args.episodes} ({reached_target/args.episodes*100:.1f}%)")

    # Calculate score percentage according to formula
    capped_score = min(mean_score, 19)
    if args.task == "2":
        score_percentage = (capped_score + 21) / 40 * 20
        print(f"\nScore Percentage (Task 2 formula): {score_percentage:.2f}% (out of 20%)")
    else:
        # Task 3 uses different grading based on when score 19 was reached
        print(f"\nTask 3 grading: Based on when score 19 was first reached during training")
        print(f"Current evaluation mean score: {mean_score:.2f}")

    # Print summary for report
    print("\n" + "=" * 60)
    print("SUMMARY FOR REPORT")
    print("=" * 60)
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Mean Score over {args.episodes} episodes: {mean_score:.2f}")
    print(f"Standard Deviation: {std_score:.2f}")

    # Save results to text file
    results_path = os.path.join(args.output_dir, f"task{args.task}_evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"DQN Evaluation Results - Task {args.task}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Date: {os.popen('date').read().strip()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of episodes: {args.episodes}\n")
        f.write(f"Seeds used: {seeds[0]} to {seeds[-1]}\n\n")
        f.write("Episode Details:\n")
        f.write("-" * 40 + "\n")
        for i, detail in enumerate(episode_details):
            f.write(f"Episode {i+1}: Seed={detail['seed']}, Score={detail['score']}, "
                   f"Total Reward={detail['total_reward']:.1f}, Steps={detail['steps']}\n")
        f.write("\n" + "-" * 40 + "\n")
        f.write(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}\n")
        f.write(f"Min Score: {min_score}\n")
        f.write(f"Max Score: {max_score}\n")
        f.write(f"Episodes reaching score 19: {reached_target}/{args.episodes}\n")

    print(f"\nResults saved to: {results_path}")

    plot_path = os.path.join(args.output_dir, f"task{args.task}_evaluation_plot.png")
    plot_evaluation_results(all_scores, seeds, args.task, plot_path)

    print("\n" + "=" * 60)
    print("INCLUDE THIS SCREENSHOT IN YOUR REPORT:")
    print("=" * 60)
    print(f"Mean reward of {args.episodes} evaluation episodes: {mean_score:.2f}")
    print(f"This screenshot should show the evaluation results across seeds {seeds[0]}-{seeds[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN on Pong-v5")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained .pt model")
    parser.add_argument("--task", type=str, default="2", choices=["2", "3"],
                        help="Task number (2 or 3)")
    parser.add_argument("--output-dir", type=str, default="./eval_videos",
                        help="Directory to save videos and plots")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20 as per protocol)")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save videos for each episode")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save keyframes as screenshots")

    args = parser.parse_args()
    evaluate(args)