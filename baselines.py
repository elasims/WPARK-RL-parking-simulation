"""
Baseline agents for benchmarking the RL policy.

RandomAgent   — chooses a random free space (or rejects if full).
GreedyAgent   — always assigns the free space closest to the car's target shop.
"""

import numpy as np
from environment import (
    idx_to_space, proximity_score, TOTAL_SPACES, SHOP_TARGET, FLOORS
)


class RandomAgent:
    """Uniformly random free-space assignment."""

    def predict(self, obs: np.ndarray, env) -> int:
        free = np.where(env.occupancy == 0)[0]
        if len(free) == 0:
            return TOTAL_SPACES   # reject
        return int(np.random.choice(free))


class GreedyAgent:
    """
    Picks the free space with the best proximity score to the car's target.
    Breaks ties by preferring lower floor numbers (less driving).
    Immediately assigns ground-floor close bays for medical cars.
    """

    def predict(self, obs: np.ndarray, env) -> int:
        car  = env.current_car
        free = np.where(env.occupancy == 0)[0]
        if len(free) == 0:
            return TOTAL_SPACES

        tf, tz = SHOP_TARGET[car["shop"]]

        # Medical: force ground floor if any free space there
        if car["urgency"] == "Medical":
            f1_free = [i for i in free if idx_to_space(i)[0] == 1]
            if f1_free:
                free = np.array(f1_free)

        best_idx  = free[0]
        best_prox = 999
        best_floor = 999

        for i in free:
            af, az, _ = idx_to_space(i)
            p = proximity_score(af, az, tf, tz)
            if p < best_prox or (p == best_prox and af < best_floor):
                best_prox  = p
                best_floor = af
                best_idx   = i

        return int(best_idx)
