import argparse
import os
import random
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

PONG_BEST_REWARD = -21
STATE_SHAPE = (4, 84, 84)

gym.register_envs(ale_py)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def make_transition(state, action, reward, next_state, done):
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
    }


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
        stacked = np.stack(self.frames, axis=0)
        return stacked


class SumTree:
    """SumTree with pre-allocated numpy arrays for structured data storage.

    Instead of storing Python objects, data is stored in contiguous numpy
    arrays. This makes batch indexing O(1) via fancy indexing instead of
    O(batch_size) Python-level list comprehensions + np.array stacking.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.states = np.zeros((capacity, *STATE_SHAPE), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *STATE_SHAPE), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, delta: float):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += delta
            parent //= 2

    def _leaf_index(self, data_idx: int) -> int:
        return data_idx + self.capacity

    @property
    def total(self) -> float:
        return self.tree[1]

    def add(self, priority: float, state, action, reward, next_state, done):
        leaf_idx = self._leaf_index(self.write_ptr)
        delta = priority - self.tree[leaf_idx]

        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

        self.states[self.write_ptr] = state
        self.actions[self.write_ptr] = action
        self.rewards[self.write_ptr] = reward
        self.next_states[self.write_ptr] = next_state
        self.dones[self.write_ptr] = float(done)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, data_idx: int, priority: float):
        leaf_idx = self._leaf_index(data_idx)
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def batch_get(self, values: np.ndarray):
        batch_size = len(values)
        data_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            s = values[i]
            idx = 1
            while idx < self.capacity:
                left = 2 * idx
                if s <= self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left]
                    idx = left + 1
            data_indices[i] = idx - self.capacity
            priorities[i] = self.tree[idx]

        return data_indices, priorities


class PrioritizedReplayBuffer:
    def __init__(
            self,
            capacity: int,
            alpha: float = 0.6,
            beta: float = 0.4,
            beta_increment: float = 1e-4,
            epsilon: float = 1e-5,
    ):
        assert capacity & (capacity - 1) == 0, "capacity must be a power of 2"

        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def _priority(self, td_error: float) -> float:
        return (abs(td_error) + self.epsilon) ** self.alpha

    def add(self, transition, td_error: float = None):
        priority = self.max_priority if td_error is None else self._priority(td_error)
        self.tree.add(
            priority,
            transition["state"],
            transition["action"],
            transition["reward"],
            transition["next_state"],
            transition["done"],
        )

    def sample(self, batch_size: int):
        self.beta = min(1.0, self.beta + self.beta_increment)

        segment = self.tree.total / batch_size
        lows = np.arange(batch_size, dtype=np.float64) * segment
        values = np.random.uniform(lows, lows + segment)

        indices, priorities = self.tree.batch_get(values)

        n = self.tree.size
        probs = priorities / self.tree.total
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()

        states = self.tree.states[indices]
        next_states = self.tree.next_states[indices]
        actions = self.tree.actions[indices]
        rewards = self.tree.rewards[indices]
        dones = self.tree.dones[indices]

        return states, actions, rewards, next_states, dones, weights.astype(np.float32), indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            p = self._priority(float(err))
            self.tree.update(int(idx), p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.size


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """

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


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.env.observation_space.seed(seed=seed)
        self.env.action_space.seed(seed=seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        print("Using device:", self.device, "| AMP:", self.use_amp)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.episodes = args.episodes
        self.max_episode_steps = args.max_episode_steps
        self.total_expected_steps = self.episodes * self.max_episode_steps
        self.clip = args.clip

        total_train_steps = self.total_expected_steps * args.train_per_step
        beta_step = (1.0 - args.bias_annealing_factor) / total_train_steps

        self.preprocessor = AtariPreprocessor()
        self.memory = PrioritizedReplayBuffer(args.memory_size, beta_step, args.priority_importance_factor,
                                              args.bias_annealing_factor)
        self.q_net = DQN(self.preprocessor.frame_stack, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.preprocessor.frame_stack, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        if hasattr(torch, "compile"):
            try:
                self.q_net = torch.compile(self.q_net, mode="reduce-overhead")
                self.target_net = torch.compile(self.target_net, mode="reduce-overhead")
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile unavailable, skipping: {e}")

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.env_count = 0
        self.train_count = 0
        self.best_reward = PONG_BEST_REWARD
        self.replay_start_size = args.replay_start_size
        self.min_buffer_to_train = max(self.replay_start_size, self.batch_size)
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True)
        state_tensor.mul_(1.0 / 255.0)
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.use_amp):
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self):
        for ep in range(self.episodes):
            obs, _ = self.env.reset(seed=seed + ep)

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                self.memory.add(make_transition(state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(
                        f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
            print(
                f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            if ep % 50 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)

        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True)
            state_tensor.mul_(1.0 / 255.0)
            with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.use_amp):
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward

    def train(self):
        if len(self.memory) < self.min_buffer_to_train:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device, dtype=torch.float32, non_blocking=True).mul_(1.0 / 255.0)
        next_states_t = torch.from_numpy(next_states).to(self.device, dtype=torch.float32, non_blocking=True).mul_(
            1.0 / 255.0)
        weights_t = torch.from_numpy(weights).to(self.device, non_blocking=True)
        actions_t = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards_t = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        dones_t = torch.from_numpy(dones).to(self.device, non_blocking=True)

        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions_indices = self.q_net(next_states_t).argmax(1)
                target_values = self.target_net(next_states_t).gather(1, next_actions_indices.unsqueeze(1)).squeeze(1)

            y = rewards_t + self.gamma * target_values * (1 - dones_t)
            td_errors = q_values - y
            loss = torch.mean((td_errors ** 2) * weights_t)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            new_errors = td_errors.abs().cpu().numpy()
        self.memory.update_priorities(indices, new_errors)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(
                f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log(
                {
                    "Loss": loss.item(),
                    "Env Step Count": self.env_count,
                    "Q Mean": q_values.mean().item(),
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results/vanilla-pong")
    parser.add_argument("--wandb-run-name", type=str, default="vanilla-pong-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=2 ** 17)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999975)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=20_000)
    parser.add_argument("--max-episode-steps", type=int, default=27_000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--bias-annealing-factor", type=float, default=0.4)
    parser.add_argument("--priority-importance-factor", type=float, default=0.6)
    parser.add_argument("--clip", default=10.0, type=float)

    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()
