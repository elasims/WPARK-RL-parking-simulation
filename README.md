# WPARK-RL-parking-simulation
# Mall Car Park — Reinforcement Learning System

## Overview

A Gymnasium-based RL environment that trains a PPO agent to assign incoming
cars to optimal parking spaces across a 3-floor, 4-zone mall car park.
The primary objective is to minimise congestion and navigation time.

---

## File Structure

```
carpark_rl/
├── environment.py    # Gymnasium env — state, action, reward
├── baselines.py      # Random and Greedy comparison agents
├── train.py          # PPO training script (Stable-Baselines3)
├── infer.py          # Load model + run live assignments
├── evaluate.py       # Head-to-head agent comparison
└── README.md
```

---

## Quick Start

### 1. Requirements

numpy
pandas
openpyxl
gymnasium
stable-baselines3
torch
tensorboard


Install with:
```bash
pip install numpy pandas openpyxl gymnasium stable-baselines3 torch tensorboard
```

### 2. Train the agent
```bash
# Synthetic data (no Excel file needed)
python train.py --timesteps 200000

# Real data from the Excel dataset
python train.py --data mall_carpark_rl_dataset.xlsx --timesteps 300000
```

### 3. Evaluate agents head-to-head
```bash
python evaluate.py --episodes 30 --ppo
```

### 4. Run inference / demo
```bash
# 30-car demo
python infer.py --render

# Single car assignment
python infer.py --car "Tesco Express" Medium Non-urgent Short

# With real data
python infer.py --data mall_carpark_rl_dataset.xlsx --n_cars 50
```

### 5. View TensorBoard training curves
```bash
tensorboard --logdir logs/
```

---

## Environment Design

### Layout
| Floor | Zones       | Shops                                               |
|-------|-------------|-----------------------------------------------------|
| 1     | North       | Tesco Express, Boots Pharmacy, Costa Coffee         |
| 1     | East        | McDonald's, Nando's, Subway, KFC                    |
| 1     | South       | Currys PC World, Apple Store, Samsung               |
| 1     | West        | Marks & Spencer, Waitrose, Holland & Barrett        |
| 2     | North       | Zara, H&M, Primark, Next                            |
| 2     | East        | Odeon Cinema, Vue Cinema                            |
| 2     | South       | JD Sports, Nike, Adidas, Foot Locker               |
| 2     | West        | LEGO Store, Smyths Toys, Build-A-Bear               |
| 3     | North       | IKEA, Dunelm, The Range                             |
| 3     | East        | Pure Gym, Fitness First                             |
| 3     | South       | B&Q, Wickes, Toolstation                            |
| 3     | West        | Staples, WHSmith, Ryman                             |

**360 total spaces** (3 floors × 4 zones × 30 slots per zone).

### Observation Vector (368 values, normalised 0–1)
| Index   | Feature                  |
|---------|--------------------------|
| 0       | Time of day (hour/23)    |
| 1–3     | Floor occupancy F1/F2/F3 |
| 4       | Shop index (purpose)     |
| 5       | Car size (0=S, 0.5=M, 1=L)|
| 6       | Urgency flag (Medical=1) |
| 7       | Duration (0=Short..1=Long)|
| 8–367   | Binary occupancy grid    |

### Action Space
`Discrete(361)` — 0–359 = assign to a specific space, 360 = reject.

### Reward Function
| Component             | Value         | Condition                          |
|-----------------------|---------------|------------------------------------|
| Proximity close       | +5.0          | proximity score ≤ 2                |
| Proximity medium      | +1.0          | proximity score == 3               |
| Proximity far         | −3.0          | proximity score ≥ 4                |
| Medical served        | +10.0         | Medical car → F1, close to exit    |
| Medical not served    | −8.0          | Medical car sent far away          |
| Congestion penalty    | −0.05 / 1%   | above 80% floor occupancy          |
| Navigation time       | −0.03 / min  | estimated drive + walk time        |
| Compact bay bonus     | +2.0          | Small car → slot < 10              |
| Turnover bonus        | +3.0          | Short-stay → North/East zone       |
| Rejection penalty     | −5.0          | Reject when free spaces exist      |
| Invalid action        | −10.0         | Assign to occupied space           |

---

## Key Metrics

- **Average navigation time** (minutes) — primary KPI
- **Average proximity score** (1–5, lower is better)
- **Medical urgency service rate**
- **Episode cumulative reward**

---

## Tuning

Edit `REWARD_CONFIG` in `environment.py` to adjust the reward weights.
Run `train.py` with different `--timesteps` values depending on convergence.
The PPO hyperparameters in `train.py` are a solid default; increase
`net_arch` to `[512, 512]` for more complex policies.

## Results

Agents evaluated head-to-head over 30 episodes (500 cars/episode) using `evaluate.py --episodes 30 --ppo`.

| Agent          | Reward   | Assigned | Rejected | Medical OK | Nav time (min) | Proximity (1–5) |
|----------------|----------|----------|----------|------------|-----------------|------------------|
| Random         | -57.11   | 500.0    | 0.0      | 3.7        | 3.29            | 3.56             |
| Greedy         | 2822.03  | 500.0    | 0.0      | 14.0       | **0.19**        | **1.16**         |
| **PPO (trained)** | **2824.43** | 499.7 | 0.3      | **14.7**   | 1.38            | 1.95             |

**Takeaways:**
- Both Greedy and PPO massively outperform Random, confirming the environment and reward signal are sound.
- PPO achieves marginally higher cumulative reward than Greedy (+0.09%) and serves more medical-urgency cars (14.7 vs 14.0 out of ~15/episode), consistent with the reward function's heavy weighting of medical priority (+10/−8).
- Greedy remains stronger on raw efficiency metrics — navigation time (0.19 vs 1.38 min) and proximity score (1.16 vs 1.95) — and has a marginally lower rejection rate (0.0 vs 0.3).
- This suggests PPO learned to trade off navigation efficiency for reward components it was optimizing more aggressively (medical service, proximity bonuses), rather than uniformly outperforming the heuristic baseline. Rebalancing `REWARD_CONFIG` — e.g. increasing the navigation-time penalty — is a natural next step to close this gap.
