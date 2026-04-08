"""
baseline/inference.py — MANDATORY entry point for the hackathon submission.

Runs the baseline agent (heuristic + optional RL) and prints structured logs.

Usage:
    python baseline/inference.py --task easy --agent heuristic
    python baseline/inference.py --task medium --agent ppo
    python baseline/inference.py --task hard --agent random
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import json
import numpy as np
import random

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import MonsoonFarmEnv
from env.models import FarmAction, CropStage, WeatherType
from env.simulator import CROP_CONFIG
from baseline.agent import HeuristicAgent
from grader.grader import get_grader

SEED = 42


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(tag: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{tag}] {ts} | {msg}")


def log_step(day: int, info: dict, render: str = ""):
    rb = info.get("reward_breakdown", {})
    log("STEP", (
        f"Day {day:3d} | "
        f"Rev: ₹{info['step_revenue_inr']:7.0f} | "
        f"Cost: ₹{info['step_cost_inr']:6.0f} | "
        f"Yield: {info['harvested_kg']:.2f}kg | "
        f"Budget: ₹{info['budget_inr']:8.0f} | "
        f"Eco: {info['eco_score']:.3f} | "
        f"Reward: {rb.get('total', 0.0):+.4f}"
    ))


# ---------------------------------------------------------------------------
# Run heuristic agent
# ---------------------------------------------------------------------------

def run_heuristic(task: str, verbose: bool = True, render_interval: int = 10):
    log("START", f"=== Monsoon Farm RL Environment ===")
    log("START", f"Task: {task.upper()} | Agent: HEURISTIC | Seed: {SEED}")

    # Build env directly (not through Gymnasium wrapper for flexibility)
    from tasks.easy   import EasyFarmEnv
    from tasks.medium import MediumFarmEnv
    from tasks.hard   import HardFarmEnv

    task_map = {"easy": EasyFarmEnv, "medium": MediumFarmEnv, "hard": HardFarmEnv}
    if task not in task_map:
        raise ValueError(f"Unknown task: {task}")

    gym_env = task_map[task](seed=SEED)
    core_env = gym_env.get_wrapped_env()

    agent = HeuristicAgent(eco_mode=True)
    grader = get_grader(task)

    obs = core_env.reset(seed=SEED)
    state = core_env.state()

    total_reward = 0.0
    step_count = 0
    episode_log = []

    log("START", f"Episode length: {state.episode_length} days | Slots: {state.num_slots}")
    log("START", f"Starting budget: ₹{state.resources.budget_inr:,.0f}")
    log("START", "-" * 70)

    done = False
    while not done:
        state = core_env.state()
        action = agent.act(state)
        obs, reward, done, info = core_env.step(action)
        total_reward += reward
        step_count += 1

        episode_log.append({
            "day": info["day"],
            "reward": round(reward, 4),
            "revenue": info["step_revenue_inr"],
            "cost": info["step_cost_inr"],
            "yield_kg": info["harvested_kg"],
            "budget": info["budget_inr"],
            "eco": info["eco_score"],
        })

        if verbose and (info["day"] % render_interval == 0 or done):
            log_step(info["day"], info)
            if render_interval <= 10:
                print(core_env.render())
                print()

    # Final grading
    final_state = core_env.state()
    grade = grader.grade(final_state)

    log("END", "=" * 70)
    log("END", f"Episode complete — {step_count} days")
    log("END", f"Total reward:       {total_reward:+.4f}")
    log("END", f"Total profit:       ₹{final_state.total_profit_inr:,.0f}")
    log("END", f"Total yield:        {final_state.total_yield_kg:.2f} kg")
    log("END", f"Final eco-score:    {final_state.eco.eco_score:.4f}")
    log("END", f"Final budget:       ₹{final_state.resources.budget_inr:,.0f}")
    log("END", "-" * 70)
    log("END", f"GRADE BREAKDOWN:")
    for k, v in grade.to_dict().items():
        log("END", f"  {k:30s}: {v}")
    log("END", f"COMPOSITE SCORE: {grade.composite_score:.4f} / 1.0")
    log("END", "=" * 70)

    return grade, episode_log


# ---------------------------------------------------------------------------
# Run PPO agent (Stable-Baselines3)
# ---------------------------------------------------------------------------

def run_ppo(task: str, train_steps: int = 50_000, verbose: bool = True):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        log("ERROR", "stable-baselines3 not installed. Run: pip install stable-baselines3")
        sys.exit(1)

    log("START", f"=== Monsoon Farm RL Environment ===")
    log("START", f"Task: {task.upper()} | Agent: PPO | Seed: {SEED}")

    from tasks.easy   import EasyFarmEnv
    from tasks.medium import MediumFarmEnv
    from tasks.hard   import HardFarmEnv
    task_map = {"easy": EasyFarmEnv, "medium": MediumFarmEnv, "hard": HardFarmEnv}

    env_cls = task_map[task]
    env = env_cls(seed=SEED)

    log("START", f"Training PPO for {train_steps} steps...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1 if verbose else 0,
        seed=SEED,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard_logs/",
    )
    model.learn(total_timesteps=train_steps)

    # Save model
    model_path = f"baseline/ppo_{task}"
    model.save(model_path)
    log("START", f"Model saved to {model_path}.zip")

    # Evaluation run
    log("START", "Running evaluation episode...")
    obs, _ = env.reset()
    grader = get_grader(task)
    core_env = env.get_wrapped_env()
    total_reward = 0.0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if verbose and info["day"] % 15 == 0:
            log_step(info["day"], info)

    final_state = core_env.state()
    grade = grader.grade(final_state)

    log("END", "=" * 70)
    log("END", f"PPO Evaluation complete")
    log("END", f"Total reward: {total_reward:+.4f}")
    log("END", f"COMPOSITE SCORE: {grade.composite_score:.4f} / 1.0")
    log("END", "=" * 70)

    return grade


# ---------------------------------------------------------------------------
# Run random agent (sanity check)
# ---------------------------------------------------------------------------

def run_random(task: str):
    log("START", f"=== Monsoon Farm RL Environment ===")
    log("START", f"Task: {task.upper()} | Agent: RANDOM | Seed: {SEED}")

    from tasks.easy   import EasyFarmEnv
    from tasks.medium import MediumFarmEnv
    from tasks.hard   import HardFarmEnv
    task_map = {"easy": EasyFarmEnv, "medium": MediumFarmEnv, "hard": HardFarmEnv}

    env = task_map[task](seed=SEED)
    core_env = env.get_wrapped_env()
    grader = get_grader(task)

    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if info["day"] % 20 == 0:
            log_step(info["day"], info)

    final_state = core_env.state()
    grade = grader.grade(final_state)

    log("END", f"Random agent — Total reward: {total_reward:+.4f}")
    log("END", f"COMPOSITE SCORE: {grade.composite_score:.4f} / 1.0")

    return grade


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monsoon Farm RL Environment — Baseline Inference"
    )
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard"], default="easy",
        help="Task difficulty level"
    )
    parser.add_argument(
        "--agent", choices=["heuristic", "ppo", "random"], default="heuristic",
        help="Agent type to run"
    )
    parser.add_argument(
        "--train-steps", type=int, default=50_000,
        help="Training steps for PPO agent"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output"
    )
    parser.add_argument(
        "--render-interval", type=int, default=15,
        help="Render farm state every N days"
    )
    parser.add_argument(
        "--save-log", type=str, default=None,
        help="Save episode log to JSON file"
    )
    args = parser.parse_args()

    if args.agent == "heuristic":
        grade, episode_log = run_heuristic(
            args.task, args.verbose, args.render_interval
        )
        if args.save_log:
            with open(args.save_log, "w") as f:
                json.dump({"grade": grade.to_dict(), "episode": episode_log}, f, indent=2)
            log("END", f"Log saved to {args.save_log}")

    elif args.agent == "ppo":
        run_ppo(args.task, args.train_steps, args.verbose)

    elif args.agent == "random":
        run_random(args.task)


if __name__ == "__main__":
    main()
