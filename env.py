"""
Post-Disaster Construction Site Environment
15 ground robots coordinate to locate and clear debris
using stigmergy-based communication (pheromone trails).
"""

import numpy as np
import random

# Cell types
EMPTY   = 0
DEBRIS  = 1
CLEARED = 2
AGENT   = 3
WALL    = 4

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),   # cardinal
              (-1, -1), (-1, 1), (1, -1), (1, 1)]   # diagonal


class DebrisClearanceEnv:
    def __init__(self, grid_size=20, n_agents=15, n_debris=40,
                 n_disabled=0, pheromone_decay=0.97):
        self.grid_size     = grid_size
        self.n_agents      = n_agents
        self.n_debris      = n_debris
        self.n_disabled    = n_disabled        # simulate agent failures
        self.pheromone_decay = pheromone_decay
        self.action_space  = 9                 # 8 directions + stay
        self.reset()

    def reset(self):
        self.grid      = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.pheromone = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # Place debris
        self.debris_positions = set()
        while len(self.debris_positions) < self.n_debris:
            r = random.randint(2, self.grid_size - 3)
            c = random.randint(2, self.grid_size - 3)
            self.debris_positions.add((r, c))
            self.grid[r][c] = DEBRIS

        self.total_debris = len(self.debris_positions)
        self.cleared      = 0

        # Place agents along the bottom edge (entry point)
        self.agents = []
        for i in range(self.n_agents):
            disabled = (i < self.n_disabled)
            c = (i * (self.grid_size // self.n_agents)) % self.grid_size
            self.agents.append({
                "id":       i,
                "pos":      [self.grid_size - 1, c],
                "disabled": disabled,
                "carrying": False
            })

        self.steps    = 0
        self.max_steps = 400
        return self._get_obs()

    def _get_obs(self):
        """Each agent observes its local 5x5 neighbourhood + pheromone intensity."""
        obs_list = []
        for agent in self.agents:
            r, c   = agent["pos"]
            local  = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        local.append(self.grid[nr][nc] / 4.0)
                        local.append(self.pheromone[nr][nc])
                    else:
                        local.append(1.0)   # treat out-of-bounds as wall
                        local.append(0.0)
            obs_list.append(np.array(local, dtype=np.float32))
        return obs_list

    def step(self, actions):
        self.steps += 1
        rewards = []

        for i, agent in enumerate(self.agents):
            if agent["disabled"]:
                rewards.append(0.0)
                continue

            action = actions[i]
            r, c   = agent["pos"]

            if action < 8:
                dr, dc = DIRECTIONS[action]
                nr, nc = r + dr, c + dc

                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                        and self.grid[nr][nc] != WALL):
                    agent["pos"] = [nr, nc]
                    r, c = nr, nc

            reward = -0.01  # small step penalty

            # Deposit pheromone at current position
            self.pheromone[r][c] = min(1.0, self.pheromone[r][c] + 0.3)

            # Clear debris if standing on it
            if self.grid[r][c] == DEBRIS:
                self.grid[r][c] = CLEARED
                self.debris_positions.discard((r, c))
                self.cleared += 1
                reward = 5.0   # big reward for clearing debris

            rewards.append(reward)

        # Decay pheromones (stigmergy evaporation)
        self.pheromone *= self.pheromone_decay

        done       = (self.steps >= self.max_steps or self.cleared == self.total_debris)
        completion = self.cleared / self.total_debris if self.total_debris > 0 else 1.0

        return self._get_obs(), rewards, done, {
            "cleared":    self.cleared,
            "total":      self.total_debris,
            "completion": completion,
            "steps":      self.steps
        }

    def render(self):
        display = self.grid.copy().astype(object)
        display[display == EMPTY]   = '.'
        display[display == DEBRIS]  = 'D'
        display[display == CLEARED] = '*'
        display[display == WALL]    = '#'

        for agent in self.agents:
            r, c = agent["pos"]
            display[r][c] = 'F' if agent["disabled"] else 'R'

        active  = sum(1 for a in self.agents if not a["disabled"])
        print(f"\nStep: {self.steps}  |  Cleared: {self.cleared}/{self.total_debris} "
              f"({self.cleared/self.total_debris*100:.1f}%)  |  Active robots: {active}/{self.n_agents}")
        print("  " + "".join([str(i % 10) for i in range(self.grid_size)]))
        for i, row in enumerate(display):
            print(f"{i:2d} " + "".join(row.astype(str)))
        print("Legend: . = empty, D = debris, * = cleared, R = robot, F = failed robot")
