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
from torch.nn import MSELoss

PONG_BEST_REWARD = -21

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


class MultiStepHandler:
    def __init__(self, step_cnt, gamma):
        self._step_cnt = step_cnt
        self._gamma = gamma
        self.window: deque[dict] = deque(maxlen=step_cnt)

    def append(self, transition: dict) -> list[dict] | None:
        self.window.append(transition)
        if transition["done"]:
            return self._flush()
        if len(self.window) >= self._step_cnt:
            return [self._transform_first_transition()]
        return None

    def _flush(self):
        transitions = []
        while len(self.window) > 0:
            transitions.append(self._transform_first_transition())
        return transitions

    def _transform_first_transition(self):
        seq_reward = self._compute_seq_reward()
        first_trans = self.window.popleft()
        last_trans = self.window[-1] if self.window else first_trans
        first_trans["reward"] = seq_reward
        first_trans["next_state"] = last_trans["next_state"]
        first_trans["done"] = last_trans["done"]
        return first_trans

    def _compute_seq_reward(self):
        reward_sum = 0
        for i, trans in enumerate(self.window):
            reward_sum += trans["reward"] * (self._gamma ** i)
        return reward_sum


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


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def append(self, transition: dict):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        tup = (
            transition["state"],
            transition["action"],
            transition["reward"],
            transition["next_state"],
            transition["done"],
        )
        self.buffer.append(tup)


    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


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
        print("Using device:", self.device)

        self.preprocessor = AtariPreprocessor()
        self.multi_step_handler = MultiStepHandler(args.reward_step_count, args.discount_factor)
        self.memory = ReplayBuffer(args.memory_size)
        self.q_net = DQN(self.preprocessor.frame_stack, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.preprocessor.frame_stack, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.criterion = MSELoss()

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.reward_step_count = args.reward_step_count
        self.episodes = args.episodes

        self.env_count = 0
        self.train_count = 0
        self.best_reward = PONG_BEST_REWARD
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.min_buffer_to_train = max(self.replay_start_size, self.batch_size)
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
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
                transitions = self.multi_step_handler.append(make_transition(state, action, reward, next_state, done))
                if transitions is not None:
                    for trans in transitions:
                        self.memory.append(trans)

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
            if ep % 100 == 0:
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
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
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

        (states, actions, rewards, next_states, dones) = self.memory.sample(self.batch_size)


        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions_indices = self.q_net(next_states).argmax(1)
            target_values = self.target_net(next_states).gather(1, next_actions_indices.unsqueeze(1)).squeeze(1)

        y = rewards + (self.gamma ** self.reward_step_count) * target_values * (1 - dones)
        loss = self.criterion(q_values, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

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
    parser.add_argument("--memory-size", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=27_000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--reward-step-count", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1500)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()
