import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = []
        for _ in range(batch_size):
            batch.append(random.choice(self.buffer))
        return batch

    def __len__(self):
        return len(self.buffer)