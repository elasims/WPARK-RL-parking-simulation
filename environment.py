"""
Mall Car Park RL Environment
============================
A Gymnasium environment that simulates a 3-floor, 4-zone mall car park.
The agent receives an incoming car's attributes and must assign it to the
optimal parking space, minimising congestion and navigation time.

State space  : occupancy grid + floor load + time-of-day
Observation  : car purpose, size, urgency, duration + environment state
Action space : discrete — assign to one of N spaces (or reject)
Reward       : composite (proximity + urgency + congestion + turnover)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import random


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOORS           = [1, 2, 3]
ZONES            = ["North", "East", "South", "West"]
SPACES_PER_ZONE  = 30          # 30 spaces per zone → 120 per floor → 360 total
TOTAL_SPACES     = len(FLOORS) * len(ZONES) * SPACES_PER_ZONE   # 360

CAR_SIZES  = ["Small", "Medium", "Large"]
DURATIONS  = ["Short", "Medium", "Long"]
URGENCIES  = ["Non-urgent", "Medical"]

# Shops per (floor, zone) — mirrors the Excel directory
SHOP_MAP = {
    (1, "North"): ["Tesco Express", "Boots Pharmacy", "Costa Coffee"],
    (1, "East"):  ["McDonald's", "Nando's", "Subway", "KFC"],
    (1, "South"): ["Currys PC World", "Apple Store", "Samsung"],
    (1, "West"):  ["Marks & Spencer", "Waitrose", "Holland & Barrett"],
    (2, "North"): ["Zara", "H&M", "Primark", "Next"],
    (2, "East"):  ["Odeon Cinema", "Vue Cinema"],
    (2, "South"): ["JD Sports", "Nike", "Adidas", "Foot Locker"],
    (2, "West"):  ["LEGO Store", "Smyths Toys", "Build-A-Bear"],
    (3, "North"): ["IKEA", "Dunelm", "The Range"],
    (3, "East"):  ["Pure Gym", "Fitness First"],
    (3, "South"): ["B&Q", "Wickes", "Toolstation"],
    (3, "West"):  ["Staples", "WHSmith", "Ryman"],
}

ALL_SHOPS    = [shop for shops in SHOP_MAP.values() for shop in shops]
SHOP_TO_IDX  = {s: i for i, s in enumerate(ALL_SHOPS)}

# Map shop → ideal (floor, zone)
SHOP_TARGET: dict[str, tuple[int, str]] = {}
for (fl, zone), shops in SHOP_MAP.items():
    for shop in shops:
        SHOP_TARGET[shop] = (fl, zone)

# Reward weights (tune these during training)
REWARD_CONFIG = {
    "proximity_close":     +5.0,   # proximity score 1-2
    "proximity_medium":    +1.0,   # proximity score 3
    "proximity_far":       -3.0,   # proximity score 4-5
    "medical_served":     +10.0,   # medical car → floor 1 + close to exit
    "medical_not_served":  -8.0,   # medical car sent far away
    "congestion_penalty":  -0.05,  # per 1% above 80% floor occupancy
    "nav_time_penalty":    -0.03,  # per extra minute of estimated drive time
    "size_fit_bonus":      +2.0,   # small car → compact bay (slot < 10)
    "turnover_bonus":      +3.0,   # short-stay → high-turnover zone
    "rejection_penalty":   -5.0,   # reject a car when spaces still exist
    "invalid_action":     -10.0,   # tried to assign an already-occupied space
}

FLOOR_CHANGE_MINUTES = 2.0   # cost in minutes per floor transition
ZONE_DISTANCE_MINUTES = 1.5  # cost per zone step away from target


# ---------------------------------------------------------------------------
# Space index helpers
# ---------------------------------------------------------------------------

def space_idx(floor: int, zone: str, slot: int) -> int:
    """Convert (floor, zone, slot) → flat index 0..359."""
    f = FLOORS.index(floor)
    z = ZONES.index(zone)
    return f * (len(ZONES) * SPACES_PER_ZONE) + z * SPACES_PER_ZONE + slot


def idx_to_space(idx: int) -> tuple[int, str, int]:
    """Convert flat index → (floor, zone, slot)."""
    slots_per_floor = len(ZONES) * SPACES_PER_ZONE
    f    = idx // slots_per_floor
    rem  = idx % slots_per_floor
    z    = rem // SPACES_PER_ZONE
    slot = rem % SPACES_PER_ZONE
    return FLOORS[f], ZONES[z], slot


def zone_distance(z1: str, z2: str) -> int:
    """Circular Manhattan distance between two zones (0-2)."""
    pos = {"North": 0, "East": 1, "South": 2, "West": 3}
    d = abs(pos[z1] - pos[z2])
    return min(d, 4 - d)   # wrap-around


def proximity_score(af: int, az: str, tf: int, tz: str) -> int:
    """Score 1 (ideal) – 5 (far). Combines floor and zone distances."""
    floor_diff = abs(af - tf)
    zone_diff  = zone_distance(az, tz)
    return min(5, max(1, floor_diff * 2 + zone_diff + 1))


def estimated_nav_time(af: int, az: str, tf: int, tz: str) -> float:
    """Estimated navigation time in minutes."""
    return (abs(af - tf) * FLOOR_CHANGE_MINUTES
            + zone_distance(az, tz) * ZONE_DISTANCE_MINUTES)


# ---------------------------------------------------------------------------
# Car generator — real-data-driven or synthetic
# ---------------------------------------------------------------------------

class CarGenerator:
    """
    Generates incoming car events.
    If a DataFrame from the real dataset is supplied, samples from it
    (looping). Otherwise generates synthetic cars with realistic distributions.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df  = None
        self.idx = 0
        if df is not None:
            required = {"Purpose (Shop)", "Car_Dimensions", "Urgency", "Duration_Category"}
            if required.issubset(df.columns):
                self.df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    def next(self, hour: int = 12) -> dict:
        if self.df is not None:
            row     = self.df.iloc[self.idx % len(self.df)]
            self.idx += 1
            shop    = row.get("Purpose (Shop)",  random.choice(ALL_SHOPS))
            size    = row.get("Car_Dimensions",  "Medium")
            urgency = row.get("Urgency",         "Non-urgent")
            dur     = row.get("Duration_Category", "Medium")

            # Sanitise values that might not appear in our lookup tables
            shop    = shop    if shop    in SHOP_TARGET else random.choice(ALL_SHOPS)
            size    = size    if size    in CAR_SIZES   else "Medium"
            urgency = urgency if urgency in URGENCIES   else "Non-urgent"
            # Duration column may have "Short (<1h)" etc. — normalise
            dur_map = {"Short": "Short", "Medium": "Medium", "Long": "Long"}
            dur     = next((v for k, v in dur_map.items() if str(dur).startswith(k)), "Medium")
        else:
            shop    = random.choice(ALL_SHOPS)
            size    = random.choices(CAR_SIZES, weights=[0.30, 0.50, 0.20])[0]
            urgency = random.choices(URGENCIES, weights=[0.92, 0.08])[0]
            dur     = random.choices(DURATIONS, weights=[0.35, 0.45, 0.20])[0]

        return {"shop": shop, "size": size, "urgency": urgency,
                "duration": dur, "hour": hour}


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class MallCarParkEnv(gym.Env):
    """
    Observation vector (flat, normalised to 0-1):
        [0]       hour / 23                         (time of day)
        [1:4]     floor occupancy rates F1,F2,F3
        [4]       shop index / n_shops              (purpose)
        [5]       car size index / 2                (0=Small .. 1=Large)
        [6]       urgency flag                      (0=Non-urgent, 1=Medical)
        [7]       duration index / 2                (0=Short .. 1=Long)
        [8:368]   binary occupancy grid             (360 values)

    Action:
        Discrete(TOTAL_SPACES + 1)
        0 .. 359  → assign car to that space
        360       → reject (redirect to overflow / exit)

    Episode:
        One episode = n_steps arriving cars (simulating one trading day).
        Departures happen stochastically after each step.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: Optional[pd.DataFrame]  = None,
        n_steps: int                = 500,
        render_mode: Optional[str]  = None,
        reward_config: Optional[dict] = None,
    ):
        super().__init__()
        self.n_steps     = n_steps
        self.render_mode = render_mode
        self.rc          = reward_config or REWARD_CONFIG
        self.generator   = CarGenerator(df)

        # Action: 0..359 = space, 360 = reject
        self.action_space = spaces.Discrete(TOTAL_SPACES + 1)

        obs_size = 8 + TOTAL_SPACES
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self._init_state()

    # ------------------------------------------------------------------ state

    def _init_state(self):
        self.occupancy      = np.zeros(TOTAL_SPACES, dtype=np.float32)
        self.step_count     = 0
        self.hour           = 8
        self.current_car    = None
        self.episode_reward = 0.0
        self.stats = {
            "assigned": 0, "rejected": 0, "medical_served": 0,
            "invalid":  0, "total_nav_time": 0.0, "proximity_scores": [],
        }

    def _floor_occupancy(self) -> np.ndarray:
        slots = len(ZONES) * SPACES_PER_ZONE
        return np.array([
            self.occupancy[i * slots:(i + 1) * slots].mean()
            for i in range(len(FLOORS))
        ], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        car  = self.current_car
        focc = self._floor_occupancy()
        return np.concatenate([
            [self.hour / 23.0],
            focc,
            [SHOP_TO_IDX[car["shop"]] / max(len(ALL_SHOPS) - 1, 1)],
            [CAR_SIZES.index(car["size"])  / 2.0],
            [float(car["urgency"] == "Medical")],
            [DURATIONS.index(car["duration"]) / 2.0],
            self.occupancy,
        ]).astype(np.float32)

    # ----------------------------------------------------------------- gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        self.current_car = self.generator.next(self.hour)
        return self._obs(), {}

    def step(self, action: int):
        car    = self.current_car
        reward = 0.0
        info   = {"step": self.step_count}
        tf, tz = SHOP_TARGET[car["shop"]]

        if action == TOTAL_SPACES:          # ---- REJECT ----
            free = int((self.occupancy == 0).sum())
            if free > 0:
                reward += self.rc["rejection_penalty"]
            self.stats["rejected"] += 1
            info["event"] = "rejected"

        else:                               # ---- ASSIGN ----
            if self.occupancy[action] == 1:
                reward += self.rc["invalid_action"]
                self.stats["invalid"] += 1
                info["event"] = "invalid"
            else:
                af, az, slot = idx_to_space(action)
                prox     = proximity_score(af, az, tf, tz)
                nav_time = estimated_nav_time(af, az, tf, tz)
                focc     = self._floor_occupancy()

                # Proximity
                if prox <= 2:
                    reward += self.rc["proximity_close"]
                elif prox == 3:
                    reward += self.rc["proximity_medium"]
                else:
                    reward += self.rc["proximity_far"]

                # Navigation time
                reward += nav_time * self.rc["nav_time_penalty"]

                # Medical priority
                if car["urgency"] == "Medical":
                    if af == 1 and prox <= 2:
                        reward += self.rc["medical_served"]
                        self.stats["medical_served"] += 1
                    else:
                        reward += self.rc["medical_not_served"]

                # Congestion on assigned floor
                occ_pct = focc[af - 1] * 100
                if occ_pct > 80:
                    reward += (occ_pct - 80) * self.rc["congestion_penalty"]

                # Compact bay bonus for small cars
                if car["size"] == "Small" and slot < 10:
                    reward += self.rc["size_fit_bonus"]

                # Turnover bonus: short-stay → North/East (near entrance)
                if car["duration"] == "Short" and az in ("North", "East"):
                    reward += self.rc["turnover_bonus"]

                self.occupancy[action] = 1.0
                self.stats["assigned"]       += 1
                self.stats["total_nav_time"] += nav_time
                self.stats["proximity_scores"].append(prox)
                info.update({"event": "assigned", "floor": af, "zone": az,
                             "proximity": prox, "nav_time": round(nav_time, 2)})

        # Stochastic departures (~15% of occupied spaces leave each step)
        occ_idx = np.where(self.occupancy == 1)[0]
        if len(occ_idx):
            n_depart = max(1, int(len(occ_idx) * 0.15))
            self.occupancy[np.random.choice(occ_idx, n_depart, replace=False)] = 0.0

        # Advance simulated clock (08:00 → 22:00 over episode)
        self.step_count     += 1
        self.hour            = 8 + int(self.step_count / self.n_steps * 14)
        self.episode_reward += reward
        self.current_car     = self.generator.next(self.hour)

        info["episode_reward"] = self.episode_reward
        terminated = self.step_count >= self.n_steps

        return self._obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode not in ("human", "ansi"):
            return
        focc = self._floor_occupancy()
        car  = self.current_car
        bar  = lambda v: "█" * int(v * 10) + "░" * (10 - int(v * 10))
        print(
            f"\n{'─'*54}\n"
            f"  Step {self.step_count:>4}/{self.n_steps}   {self.hour:02d}:00\n"
            f"  Car   : {car['shop']:<22} | {car['size']:<6} | {car['urgency']}\n"
            f"  F1 [{bar(focc[0])}] {focc[0]:.0%}\n"
            f"  F2 [{bar(focc[1])}] {focc[1]:.0%}\n"
            f"  F3 [{bar(focc[2])}] {focc[2]:.0%}\n"
            f"  Free  : {int((self.occupancy==0).sum())}/{TOTAL_SPACES}   "
            f"Reward so far: {self.episode_reward:+.1f}\n"
            f"{'─'*54}"
        )

    def close(self):
        pass

    def summary(self) -> dict:
        s = self.stats
        n = max(s["assigned"], 1)
        return {
            "total_steps":      self.step_count,
            "assigned":         s["assigned"],
            "rejected":         s["rejected"],
            "invalid_actions":  s["invalid"],
            "medical_served":   s["medical_served"],
            "avg_nav_time_min": round(s["total_nav_time"] / n, 2),
            "avg_proximity":    round(float(np.mean(s["proximity_scores"])), 2)
                                if s["proximity_scores"] else None,
            "episode_reward":   round(self.episode_reward, 2),
        }
