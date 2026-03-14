"""
Train and simulate swarm debris clearance.
Tests both normal operation and fault-tolerance (agents disabled).

Run: python simulate.py
"""

import numpy as np
import matplotlib.pyplot as plt
from env import DebrisClearanceEnv
from agent import SwarmAgent

N_EPISODES   = 500
N_AGENTS     = 15
RENDER_EVERY = 200


def run_experiment(n_disabled=0, label="Normal"):
    env    = DebrisClearanceEnv(n_agents=N_AGENTS, n_disabled=n_disabled)
    agents = [SwarmAgent(agent_id=i) for i in range(N_AGENTS)]

    completions = []

    for ep in range(1, N_EPISODES + 1):
        obs_list = env.reset()
        done     = False

        while not done:
            actions  = [agents[i].choose_action(obs_list[i]) for i in range(N_AGENTS)]
            next_obs, rewards, done, info = env.step(actions)

            for i in range(N_AGENTS):
                if not env.agents[i]["disabled"]:
                    agents[i].update(obs_list[i], actions[i], rewards[i], next_obs[i], done)

            obs_list = next_obs

        for agent in agents:
            agent.decay_epsilon()

        completions.append(info["completion"])

        if ep % RENDER_EVERY == 0:
            env.render()
            print(f"[{label}] Episode {ep}/{N_EPISODES}  |  "
                  f"Completion: {info['completion']*100:.1f}%  |  "
                  f"Steps: {info['steps']}\n")

    avg = np.mean(completions[-50:]) * 100
    print(f"\n[{label}] Final avg completion (last 50 eps): {avg:.1f}%")
    return completions


print("=" * 60)
print("Experiment 1: All 15 agents active (Normal Operation)")
print("=" * 60)
comp_normal = run_experiment(n_disabled=0, label="Normal")

print("\n" + "=" * 60)
print("Experiment 2: 4 agents disabled (Fault Tolerance Test)")
print("=" * 60)
comp_fault = run_experiment(n_disabled=4, label="4 Disabled")

# ── Plot comparison ───────────────────────────────────────
window = 20
s_normal = np.convolve(comp_normal, np.ones(window)/window, mode='valid')
s_fault  = np.convolve(comp_fault,  np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(s_normal * 100, label='All 15 robots active',   color='steelblue', linewidth=2)
plt.plot(s_fault  * 100, label='4 robots disabled',      color='darkorange', linewidth=2, linestyle='--')
plt.axhline(y=75, color='red', linestyle=':', label='75% completion target')
plt.title("Swarm Debris Clearance — Fault Tolerance Comparison")
plt.xlabel("Episode")
plt.ylabel("Task Completion (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("swarm_results.png", dpi=150)
plt.show()
print("\nResults saved to swarm_results.png")
