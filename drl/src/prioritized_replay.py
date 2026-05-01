import numpy as np


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return

