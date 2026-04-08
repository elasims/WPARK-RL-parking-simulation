"""
infer.py — Load a trained PPO model and run live inference
===========================================================
Usage:
    python infer.py                            # interactive demo
    python infer.py --render                   # show car park state each step
    python infer.py --car "Tesco Express" Small Non-urgent Short
"""

import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from environment import (
    MallCarParkEnv, ALL_SHOPS, CAR_SIZES, URGENCIES, DURATIONS,
    SHOP_TARGET, idx_to_space, TOTAL_SPACES
)


MODEL_PATH   = "models/ppo_carpark_final.zip"
VECNORM_PATH = "models/vec_normalize.pkl"


def load_model_and_env(df=None, n_steps=500):
    """Load trained PPO + VecNormalize wrapper."""
    env_fn  = lambda: MallCarParkEnv(df=df, n_steps=n_steps)
    vec_env = DummyVecEnv([env_fn])

    if os.path.exists(VECNORM_PATH):
        vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("Warning: VecNormalize stats not found — using raw observations.")

    model = PPO.load(MODEL_PATH, env=vec_env)
    return model, vec_env


def assign_car(model, vec_env, shop: str, size: str,
               urgency: str, duration: str) -> dict:
    """
    Assign a single car using the trained policy.
    Returns a dict with the assigned space and key metrics.
    """
    env = vec_env.envs[0]

    # Override the current car with the requested attributes
    env.current_car = {
        "shop": shop, "size": size,
        "urgency": urgency, "duration": duration, "hour": env.hour
    }

    raw_obs = env._obs()
    obs     = vec_env.normalize_obs(raw_obs[np.newaxis, :])
    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])

    if action == TOTAL_SPACES:
        return {"decision": "rejected", "reason": "overflow / policy chose reject"}

    if env.occupancy[action] == 1:
        return {"decision": "invalid", "action": action}

    af, az, slot = idx_to_space(action)
    tf, tz       = SHOP_TARGET[shop]

    from environment import proximity_score, estimated_nav_time
    prox     = proximity_score(af, az, tf, tz)
    nav_time = estimated_nav_time(af, az, tf, tz)

    # Apply to environment
    env.occupancy[action] = 1.0

    return {
        "decision":        "assigned",
        "space_id":        f"F{af}-{az[0]}{slot+1:03d}",
        "floor":           af,
        "zone":            az,
        "slot":            slot + 1,
        "target_floor":    tf,
        "target_zone":     tz,
        "proximity_score": prox,
        "est_walk_min":    round(nav_time, 1),
        "shop":            shop,
        "urgency":         urgency,
    }


def run_demo(model, vec_env, n_cars: int = 20, render: bool = False):
    """Run a short demo episode and print each assignment."""
    env = vec_env.envs[0]
    obs_raw, _ = env.reset()
    obs = vec_env.normalize_obs(obs_raw[np.newaxis, :])

    print(f"\n{'═'*60}")
    print(f"  DEMO — {n_cars} incoming cars")
    print(f"{'═'*60}")

    for i in range(n_cars):
        car = env.current_car
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])

        obs_raw, reward, terminated, truncated, info = env.step(action)
        obs = vec_env.normalize_obs(obs_raw[np.newaxis, :])

        if action == TOTAL_SPACES:
            decision = "REJECT"
            detail   = ""
        elif info.get("event") == "invalid":
            decision = "INVALID"
            detail   = ""
        else:
            af, az, slot = idx_to_space(action)
            decision = f"F{af}-{az[0]}{slot+1:03d}"
            detail   = (f"prox={info.get('proximity','?')}  "
                        f"nav={info.get('nav_time','?')}min  "
                        f"R={reward:+.1f}")

        urg_tag = " 🚨" if car["urgency"] == "Medical" else ""
        print(
            f"  #{i+1:>2}  {car['shop']:<22} | {car['size']:<6} | "
            f"{car['duration']:<6}{urg_tag:<3} → {decision:<14} {detail}"
        )

        if render:
            env.render()

        if terminated or truncated:
            break

    print(f"\n{'─'*60}")
    s = env.summary()
    print(f"  Assigned:         {s['assigned']}")
    print(f"  Rejected:         {s['rejected']}")
    print(f"  Avg nav time:     {s['avg_nav_time_min']} min")
    print(f"  Avg proximity:    {s['avg_proximity']} / 5")
    print(f"  Episode reward:   {s['episode_reward']:+.1f}")
    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Mall Car Park — inference")
    parser.add_argument("--data",    type=str,  default=None)
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--n_cars",  type=int,  default=30)
    parser.add_argument("--car",     type=str,  default=None,
                        help="Shop name (quoted), then size urgency duration")
    parser.add_argument("shop_args", nargs="*")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}.")
        print("Run:  python train.py --timesteps 200000")
        return

    df = None
    if args.data and os.path.exists(args.data):
        df = pd.read_excel(args.data, sheet_name="Transaction Log", header=1)

    print("Loading model …")
    model, vec_env = load_model_and_env(df=df)
    print("Model loaded.\n")

    # Single-car assignment mode
    if args.car:
        parts    = [args.car] + args.shop_args
        shop     = parts[0] if len(parts) > 0 else "Tesco Express"
        size     = parts[1] if len(parts) > 1 else "Medium"
        urgency  = parts[2] if len(parts) > 2 else "Non-urgent"
        duration = parts[3] if len(parts) > 3 else "Medium"

        vec_env.envs[0].reset()
        result = assign_car(model, vec_env, shop, size, urgency, duration)
        print("Assignment result:")
        for k, v in result.items():
            print(f"  {k:<20}: {v}")
        return

    # Demo mode
    run_demo(model, vec_env, n_cars=args.n_cars, render=args.render)


if __name__ == "__main__":
    main()
