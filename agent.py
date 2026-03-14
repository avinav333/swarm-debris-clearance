"""
Stigmergy-based Swarm Agent with Probabilistic Task Allocation
Robots follow pheromone gradients to debris and use random walk
when no pheromone signal is present — mimicking ant colony behaviour.
"""

import numpy as np
import random


class SwarmAgent:
    def __init__(self, agent_id, action_space=9,
                 alpha=0.15, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.998, epsilon_min=0.05):
        self.agent_id      = agent_id
        self.action_space  = action_space
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table       = {}

    def _hash(self, obs):
        return tuple((obs * 8).astype(int))

    def choose_action(self, obs):
        if random.random() < self.epsilon:
            # Stigmergy bias: prefer directions with higher pheromone
            # obs has 50 elements: 25 cells × (grid_val, pheromone)
            pheromones = obs[1::2][:8]   # first 8 neighbours' pheromone values
            if pheromones.sum() > 0.1:
                # Probabilistically follow pheromone gradient
                probs = pheromones / pheromones.sum()
                return int(np.random.choice(len(pheromones), p=probs))
            return random.randint(0, self.action_space - 1)

        state = self._hash(obs)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, done):
        s  = self._hash(obs)
        ns = self._hash(next_obs)
        if s  not in self.q_table: self.q_table[s]  = np.zeros(self.action_space)
        if ns not in self.q_table: self.q_table[ns] = np.zeros(self.action_space)

        target = reward if done else reward + self.gamma * np.max(self.q_table[ns])
        self.q_table[s][action] += self.alpha * (target - self.q_table[s][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
