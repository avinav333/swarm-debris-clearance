# Decentralised Swarm Path Planning for Debris Clearance

A stigmergy-based swarm simulation where 15 autonomous ground robots coordinate to locate and clear debris in a post-disaster construction environment — with built-in fault tolerance against agent failures.

## Project Structure
```
swarm-debris-clearance/
├── env.py          # 20×20 grid environment with pheromone system
├── agent.py        # Stigmergy-based Q-Learning swarm agent
├── simulate.py     # Training + fault tolerance experiments
├── requirements.txt
└── README.md
```

## How It Works
- **Environment:** 20×20 grid with 40 debris zones (representing post-disaster construction rubble)
- **Coordination:** Robots deposit pheromone trails at visited cells; others follow gradients — no central controller
- **Fault Tolerance:** Tested with 4 randomly disabled agents simulating battery failure or hostile interference
- **Goal:** Maximise debris clearance completion within 400 steps

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run both experiments (normal + fault tolerance)
python simulate.py
```

## Results
| Condition | Avg Completion |
|---|---|
| All 15 robots active | ~88% |
| 4 robots disabled | ~76% |

System maintains **75%+ task completion** even with 4 agents disabled — demonstrating resilience to hostile interference or technical failures.

Comparison plot saved as `swarm_results.png`.

## Grid Legend
```
.  = empty cell
D  = debris (to be cleared)
*  = cleared debris
R  = active robot
F  = failed/disabled robot
```

## Key Concepts
- **Stigmergy:** Indirect coordination via pheromone trails (inspired by ant colonies)
- **Probabilistic task allocation:** Robots bias movement toward high-pheromone zones
- **Decentralised control:** No single point of failure — system degrades gracefully

## Author
Abhinava Mondal — B.Tech Construction Engineering, Jadavpur University  
*Aligned with Mission Sudarshan Chakra — Atmanirbhar Bharat*
