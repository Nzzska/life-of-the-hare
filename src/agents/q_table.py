from collections import defaultdict
import numpy as np

class QTableAgent:
    def __init__(self, n_actions: int, alpha=0.1, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.995):
        self.Q = defaultdict(float)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(0)

    def act(self, s):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions))
        qs = [self.Q[(s, a)] for a in range(self.n_actions)]
        return int(np.argmax(qs))

    def update(self, s, a, r, sp, done):
        best_next = 0.0 if done else max(self.Q[(sp, ap)] for ap in range(self.n_actions))
        td_target = r + self.gamma * best_next
        self.Q[(s, a)] += self.alpha * (td_target - self.Q[(s, a)])

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
