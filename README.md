#  Smart Monsoon-Resilient Urban Hydroponic Farm
### Meta PyTorch Hackathon — OpenEnv RL Environment

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29-green.svg)](https://gymnasium.farama.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  Overview

A **production-grade, OpenEnv-compatible Reinforcement Learning environment** simulating a rooftop vertical hydroponic farm in **Bengaluru, India** — one of the world's fastest-growing tech cities, where urban food security and water scarcity are pressing real-world problems.

The RL agent manages the complete crop lifecycle across a full monsoon season, making sequential decisions under realistic stochastic uncertainty.

---

##  Problem Statement

Urban rooftop farms in Bengaluru face:
- **Monsoon variability** — sudden heavy rain (June–September) vs brutal dry spells
- **Pre-monsoon heatwaves** (April–May) that stress heat-sensitive crops
- **Pest outbreaks** correlated with humidity and cloud cover
- **Volatile mandi (wholesale market) prices** with festival demand surges
- **Resource constraints** — limited water tank capacity, electricity, and budget

The agent must **maximise profit and crop yield while minimising water usage and chemical inputs** (eco-score).

---

##  Project Structure

```
monsoon_farm/
├── env/
│   ├── __init__.py
│   ├── environment.py      # OpenEnv API: reset(), step(), state()
│   ├── models.py           # Pydantic typed models (State, Action, Observation)
│   ├── simulator.py        # Weather, crop growth, pest, market simulators
│   └── reward.py           # Multi-objective reward function
├── tasks/
│   ├── __init__.py
│   ├── easy.py             # 60-day dry season, 8 slots, ₹75k budget
│   ├── medium.py           # 90-day monsoon, 12 slots, ₹50k budget
│   └── hard.py             # 120-day heatwave+monsoon, 16 slots, ₹30k budget
├── grader/
│   ├── __init__.py
│   └── grader.py           # EasyGrader, MediumGrader, HardGrader → [0.0, 1.0]
├── baseline/
│   ├── __init__.py
│   ├── agent.py            # Expert heuristic agent
│   └── inference.py        # MANDATORY entry point (--agent heuristic|ppo|random)
├── configs/
│   └── openenv.yaml        # Full OpenEnv spec
├── deployment/
│   ├── Dockerfile          # Multi-stage build (inference + Streamlit)
│   └── requirements.txt
├── app/
│   └── streamlit_app.py    # Hugging Face Spaces UI
└── README.md
```

---

##  Quickstart

### 1. Install dependencies

```bash
pip install -r deployment/requirements.txt
```

### 2. Run heuristic baseline (no training needed)

```bash
# From project root
python baseline/inference.py --task easy --agent heuristic
python baseline/inference.py --task medium --agent heuristic --render-interval 10
python baseline/inference.py --task hard --agent heuristic --save-log results_hard.json
```

### 3. Train and evaluate PPO agent

```bash
python baseline/inference.py --task medium --agent ppo --train-steps 100000
```

### 4. Random baseline (sanity check)

```bash
python baseline/inference.py --task easy --agent random
```

---

##  Environment Details

### OpenEnv API

```python
from env.environment import MonsoonFarmEnv
from env.models import FarmAction

env = MonsoonFarmEnv(num_slots=12, episode_length=90, start_month=6, seed=42)

# reset() → FarmObservation
obs = env.reset()

# step(FarmAction) → (FarmObservation, float, bool, dict)
action = env.action_space_sample()
obs, reward, done, info = env.step(action)

# state() → FarmState (full internal state)
state = env.state()

print(env.render())
```

### Gymnasium Wrappers (for SB3)

```python
from tasks.medium import MediumFarmEnv
from stable_baselines3 import PPO

env = MediumFarmEnv(seed=42)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
```

---

##  Environment Dynamics

### Weather Model (Bengaluru Climate)

| Month | Mean Temp | Avg Rain/day | Season |
|-------|-----------|--------------|--------|
| Jan-Feb | 21-23°C | 0.2-0.3mm | Dry |
| Mar-May | 26-29°C | 0.5-8mm | Pre-monsoon |
| Jun-Sep | 23-24°C | 12-15mm | **SW Monsoon** |
| Oct | 24°C | 12mm | NE Monsoon |
| Nov-Dec | 20-22°C | 0.8-4mm | Dry |

Special events:
- **Heatwaves**: 5% probability/day in April-June, +4-8°C spike
- **Heavy rain**: 15% probability during monsoon, 2-4× normal rainfall
- **Dry spells**: tracked via `consecutive_dry_days` counter

### Crop Growth Model

| Crop | Days to Harvest | Water/Day | Yield/Slot | Price (INR/kg) |
|------|----------------|-----------|------------|----------------|
| Spinach | 35 | 0.8L | 0.6kg | ₹35 |
| Lettuce | 45 | 1.0L | 0.8kg | ₹80 |
| Tomato | 75 | 1.5L | 2.5kg | ₹25 |
| Herbs | 30 | 0.5L | 0.3kg | ₹120 |

Growth stages: `EMPTY → SEEDING → JUVENILE → MATURE → HARVEST`

Health is degraded by:
- Water stress (under/over irrigation)
- Nutrient deficiency or excess
- Pest pressure
- Heat stress (heat-sensitive crops during heatwaves)

### Pest Model
- Base spread probability: 8%/day
- Amplified by humidity >75% + temperature >25°C
- Controls: biological (−10–20% pressure), chemical (−25–40% pressure)
- Chemical use degrades eco-score and risks runoff during rain

### Market Model
- Prices follow a mean-reverting random walk (±4% std/day)
- Demand multiplier: 0.5×–2.0× (festival surge during days 40-55)
- Agent can hold harvest and sell at a better price day

---

##  Tasks & Grading

### Task Comparison

| | Easy | Medium | Hard |
|-|------|--------|------|
| Episode | 60 days | 90 days | 120 days |
| Start | December | June | May |
| Slots | 8 | 12 | 16 |
| Budget | ₹75,000 | ₹50,000 | ₹30,000 |
| Action Space | MultiDiscrete | Continuous Box | Continuous Box |
| Target Profit | ₹20,000 | ₹35,000 | ₹50,000 |
| Target Yield | 30 kg | 60 kg | 100 kg |
| Target Eco | 0.6 | 0.7 | 0.8 |

### Grader Scores (all return 0.0–1.0)

```python
from grader.grader import get_grader

grader = get_grader("medium")
grade = grader.grade(env.state())

print(grade.composite_score)   # 0.0 → 1.0
print(grade.to_dict())
# {
#   "profit_score": 0.72,
#   "yield_score": 0.58,
#   "eco_score": 0.81,
#   "efficiency_score": 0.69,
#   "composite_score": 0.71,
#   "total_profit_inr": 31250.0,
#   ...
# }
```

---

##  Reward Function

```
R = 1.0 × profit_reward           (daily profit / normalisation)
  + 0.5 × yield_reward            (kg harvested / 5.0)
  + 0.3 × eco_bonus               (eco_score above 0.5)
  - 0.2 × water_penalty           (over-irrigation)
  - 0.3 × chemical_penalty        (pesticide use)
  + 0.1 × daily_health_bonus      (avg crop health × 0.1)
  + 0.2 × survival_bonus          (crops surviving extreme weather)
  - 0.5 × budget_penalty          (bankruptcy)
```

---

##  Observation Space (94 dims)

| Group | Dims | Description |
|-------|------|-------------|
| Time | 2 | day_normalized, is_monsoon |
| Weather | 8 | temp, humidity, rain, type, dry_days, 3-day forecast |
| Crop slots | 72 | 12 slots × 6 features (stage, health, nutrients, stress, pests, age) |
| Resources | 4 | water, nutrients, pesticide, budget |
| Market | 5 | 4 crop prices + demand multiplier |
| Eco/alerts | 3 | eco_score, runoff_risk, pest_alert |

---

##  Action Space

**Continuous Box** `[0, 1]^61` (medium/hard):
- Per slot (×12): `[irrigate, nutrient_dose, pest_control, harvest_flag, plant_decision]`
- Global: `[sell_fraction]`

**Discrete MultiDiscrete** `[5]^8` (easy):
- Per slot: `0=nothing, 1=plant, 2=water+feed, 3=pest_bio, 4=harvest`

---

##  Docker Deployment

```bash
# Build inference image
docker build -f deployment/Dockerfile -t monsoon-farm .

# Run heuristic agent
docker run monsoon-farm

# Run PPO agent
docker run monsoon-farm python baseline/inference.py --task medium --agent ppo

# Build Streamlit UI image
docker build --target streamlit -f deployment/Dockerfile -t monsoon-farm-ui .
docker run -p 7860:7860 monsoon-farm-ui
```

---

##  Hugging Face Spaces

The Streamlit app (`app/streamlit_app.py`) runs natively on HF Spaces:

```yaml
# spaces_config.yaml (HF Spaces)
sdk: streamlit
app_file: app/streamlit_app.py
python_version: "3.11"
```

Features:
-  Interactive step-by-step or auto-play simulation
-  Live weather dashboard
-  Visual crop slot grid (health, stage, pest pressure)
-  Plotly charts: profit, yield, eco-score, water, economics
-  Episode grading with score gauge

---

##  Dependencies

| Package | Purpose |
|---------|---------|
| `torch>=2.0` | PyTorch (Meta requirement) |
| `gymnasium>=0.29` | OpenAI Gym successor |
| `stable-baselines3>=2.1` | PPO, SAC, DQN agents |
| `pydantic>=2.0` | Typed state/action/observation models |
| `streamlit>=1.28` | HF Spaces UI |
| `plotly>=5.17` | Interactive charts |
| `numpy, scipy` | Numerical simulation |
| `pyyaml` | Config loading |

---

##  Reproducibility

All stochastic components are seeded:
```python
env = MonsoonFarmEnv(seed=42)   # Default seed
```
- Weather: `random.Random(seed)`
- Pest: `random.Random(seed)`
- Market: `random.Random(seed + 1)`

---

##  License

MIT License — see `LICENSE` for details.

---

##  Acknowledgements

- Bengaluru climate data sourced from IMD (India Meteorological Department) monthly normals
- Mandi price dynamics inspired by APMC (Agricultural Produce Market Committee) datasets
- Hydroponic crop parameters from FAO cultivation guidelines
