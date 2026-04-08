"""
tests/test_environment.py — Smoke tests for the Monsoon Farm RL environment.

Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from env.environment import MonsoonFarmEnv
from env.models import FarmAction, FarmState, FarmObservation, CropStage
from env.simulator import WeatherSimulator, CropSimulator, MarketSimulator, CROP_CONFIG
from env.reward import RewardFunction
from grader.grader import get_grader, EasyGrader, MediumGrader, HardGrader
from baseline.agent import HeuristicAgent
from tasks.easy import EasyFarmEnv
from tasks.medium import MediumFarmEnv
from tasks.hard import HardFarmEnv


SEED = 42


# ---------------------------------------------------------------------------
# Core environment tests
# ---------------------------------------------------------------------------

class TestMonsoonFarmEnv:

    def setup_method(self):
        self.env = MonsoonFarmEnv(num_slots=12, episode_length=30, seed=SEED)

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert isinstance(obs, FarmObservation)
        vec = obs.to_vector()
        assert len(vec) == self.env.obs_size

    def test_reset_state_valid(self):
        self.env.reset()
        state = self.env.state()
        assert isinstance(state, FarmState)
        assert state.day == 0
        assert len(state.crop_slots) == 12
        assert state.resources.budget_inr == 50000.0
        assert state.resources.water_tank_liters > 0

    def test_step_returns_correct_types(self):
        self.env.reset()
        action = self.env.action_space_sample()
        obs, reward, done, info = self.env.step(action)
        assert isinstance(obs, FarmObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_day(self):
        self.env.reset()
        action = self.env.action_space_sample()
        self.env.step(action)
        assert self.env.state().day == 1

    def test_episode_terminates(self):
        env = MonsoonFarmEnv(num_slots=8, episode_length=10, seed=SEED)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(env.action_space_sample())
            steps += 1
        assert steps == 10

    def test_planting_action(self):
        self.env.reset()
        action = FarmAction(
            plant_crops={0: "spinach", 1: "lettuce", 2: "tomato"},
            irrigation_levels={0: 0.5, 1: 0.5, 2: 0.5},
            nutrient_dose={0: 0.4, 1: 0.4, 2: 0.4},
        )
        self.env.step(action)
        state = self.env.state()
        assert state.crop_slots[0].crop_type == "spinach"
        assert state.crop_slots[1].crop_type == "lettuce"
        assert state.crop_slots[2].crop_type == "tomato"

    def test_invalid_crop_ignored(self):
        self.env.reset()
        action = FarmAction(plant_crops={0: "unicorn_plant"})
        self.env.step(action)
        state = self.env.state()
        assert state.crop_slots[0].crop_type == "none"

    def test_observation_vector_range(self):
        self.env.reset()
        for _ in range(5):
            action = self.env.action_space_sample()
            obs, _, _, _ = self.env.step(action)
            vec = self.env.observation_to_numpy(obs)
            assert vec.dtype == np.float32
            # Most values should be in [-1, 3] (budget can go over 1.0)
            assert np.all(np.isfinite(vec)), "Observation contains NaN/Inf"

    def test_water_tank_does_not_go_negative(self):
        self.env.reset()
        for _ in range(20):
            action = FarmAction(
                irrigation_levels={i: 1.0 for i in range(12)}
            )
            self.env.step(action)
        assert self.env.state().resources.water_tank_liters >= 0.0

    def test_eco_score_degrades_with_chemicals(self):
        env = MonsoonFarmEnv(num_slots=4, episode_length=30, seed=SEED)
        env.reset()
        # Plant first
        env.step(FarmAction(
            plant_crops={i: "spinach" for i in range(4)},
            irrigation_levels={i: 0.5 for i in range(4)},
        ))
        initial_eco = env.state().eco.eco_score
        # Heavy chemical use
        for _ in range(10):
            env.step(FarmAction(
                pest_control={i: 2 for i in range(4)},
                irrigation_levels={i: 0.5 for i in range(4)},
            ))
        final_eco = env.state().eco.eco_score
        assert final_eco <= initial_eco, "Eco score should degrade with chemical use"

    def test_render_returns_string(self):
        self.env.reset()
        render = self.env.render()
        assert isinstance(render, str)
        assert "Day" in render

    def test_full_episode_heuristic(self):
        """Run a full episode with the heuristic agent — should not crash."""
        env = MonsoonFarmEnv(num_slots=8, episode_length=60, seed=SEED)
        agent = HeuristicAgent()
        env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state = env.state()
            action = agent.act(state)
            _, reward, done, info = env.step(action)
            total_reward += reward
        assert isinstance(total_reward, float)
        assert np.isfinite(total_reward)


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestWeatherSimulator:

    def test_generates_valid_weather(self):
        sim = WeatherSimulator(seed=SEED)
        from env.models import WeatherState
        prev = WeatherState()
        for day in range(1, 91):
            w = sim.simulate_day(prev, day)
            assert 15.0 <= w.temperature_c <= 45.0
            assert 0.0 <= w.humidity_pct <= 100.0
            assert w.rainfall_mm >= 0.0
            assert 0.0 <= w.solar_radiation <= 1.0
            prev = w

    def test_monsoon_months_flagged(self):
        sim = WeatherSimulator(start_day=152, seed=SEED)  # June 1
        from env.models import WeatherState
        prev = WeatherState()
        w = sim.simulate_day(prev, 1)
        assert w.is_monsoon_season is True


class TestMarketSimulator:

    def test_prices_stay_positive(self):
        sim = MarketSimulator(seed=SEED)
        from env.models import MarketState
        market = MarketState()
        for day in range(1, 91):
            market = sim.step(market, day)
            assert market.spinach_price_inr > 0
            assert market.lettuce_price_inr > 0
            assert market.tomato_price_inr > 0
            assert market.herbs_price_inr > 0

    def test_demand_multiplier_in_range(self):
        sim = MarketSimulator(seed=SEED)
        from env.models import MarketState
        market = MarketState()
        for day in range(1, 91):
            market = sim.step(market, day)
            assert 0.5 <= market.demand_multiplier <= 2.0


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------

class TestRewardFunction:

    def test_positive_profit_gives_positive_reward(self):
        env = MonsoonFarmEnv(num_slots=4, episode_length=10, seed=SEED)
        env.reset()
        reward_fn = RewardFunction()
        state = env.state()
        rb = reward_fn.compute(
            prev_state=state,
            action=FarmAction(),
            next_state=state,
            step_revenue_inr=5000.0,
            step_cost_inr=1000.0,
            harvested_kg=2.0,
            water_used_today_l=5.0,
            pesticide_used_today=0.0,
        )
        assert rb.profit_reward > 0
        assert rb.yield_reward > 0
        assert rb.total > 0

    def test_chemical_overuse_penalised(self):
        env = MonsoonFarmEnv(num_slots=12, episode_length=10, seed=SEED)
        env.reset()
        reward_fn = RewardFunction()
        state = env.state()
        action_chem = FarmAction(pest_control={i: 2 for i in range(12)})
        rb = reward_fn.compute(
            prev_state=state,
            action=action_chem,
            next_state=state,
            step_revenue_inr=0.0,
            step_cost_inr=0.0,
            harvested_kg=0.0,
            water_used_today_l=0.0,
            pesticide_used_today=10.0,
        )
        assert rb.chemical_penalty < 0


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGraders:

    def _run_episode(self, env: MonsoonFarmEnv, agent: HeuristicAgent) -> FarmState:
        env.reset()
        done = False
        while not done:
            action = agent.act(env.state())
            _, _, done, _ = env.step(action)
        return env.state()

    def test_easy_grader_returns_valid_score(self):
        env = MonsoonFarmEnv(num_slots=8, episode_length=60, start_month=12, seed=SEED)
        agent = HeuristicAgent()
        final_state = self._run_episode(env, agent)
        grader = EasyGrader()
        grade = grader.grade(final_state)
        assert 0.0 <= grade.composite_score <= 1.0
        assert 0.0 <= grade.profit_score <= 1.0
        assert 0.0 <= grade.yield_score <= 1.0

    def test_medium_grader_returns_valid_score(self):
        env = MonsoonFarmEnv(num_slots=12, episode_length=30, start_month=6, seed=SEED)
        agent = HeuristicAgent()
        final_state = self._run_episode(env, agent)
        grader = MediumGrader()
        grade = grader.grade(final_state)
        assert 0.0 <= grade.composite_score <= 1.0

    def test_hard_grader_returns_valid_score(self):
        env = MonsoonFarmEnv(num_slots=16, episode_length=30, start_month=5, seed=SEED)
        agent = HeuristicAgent()
        final_state = self._run_episode(env, agent)
        grader = HardGrader()
        grade = grader.grade(final_state)
        assert 0.0 <= grade.composite_score <= 1.0

    def test_get_grader_factory(self):
        for difficulty in ["easy", "medium", "hard"]:
            g = get_grader(difficulty)
            assert g is not None

    def test_invalid_grader_raises(self):
        with pytest.raises(ValueError):
            get_grader("impossible")


# ---------------------------------------------------------------------------
# Gymnasium wrapper tests
# ---------------------------------------------------------------------------

class TestGymnasiumWrappers:

    def test_easy_env_gym_api(self):
        env = EasyFarmEnv(seed=SEED)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape

        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_medium_env_gym_api(self):
        env = MediumFarmEnv(seed=SEED)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        assert isinstance(reward, float)

    def test_hard_env_gym_api(self):
        env = HardFarmEnv(seed=SEED)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        assert isinstance(reward, float)

    def test_easy_env_5_episodes(self):
        env = EasyFarmEnv(seed=SEED)
        for ep in range(5):
            obs, _ = env.reset(seed=SEED + ep)
            done = False
            steps = 0
            while not done:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                assert steps <= 65, "Episode too long"
            assert steps == 60


# ---------------------------------------------------------------------------
# Heuristic agent tests
# ---------------------------------------------------------------------------

class TestHeuristicAgent:

    def test_agent_produces_valid_actions(self):
        env = MonsoonFarmEnv(num_slots=12, episode_length=10, seed=SEED)
        agent = HeuristicAgent()
        env.reset()
        for _ in range(5):
            state = env.state()
            action = agent.act(state)
            assert isinstance(action, FarmAction)
            # All irrigation values in [0, 1]
            for v in action.irrigation_levels.values():
                assert 0.0 <= v <= 1.0
            # Pest control in {0, 1, 2}
            for v in action.pest_control.values():
                assert v in (0, 1, 2)
            env.step(action)

    def test_eco_mode_avoids_chemicals(self):
        env = MonsoonFarmEnv(num_slots=12, episode_length=30, seed=SEED)
        agent = HeuristicAgent(eco_mode=True)
        # Plant and introduce high pest pressure
        env.reset()
        env.step(FarmAction(plant_crops={i: "spinach" for i in range(12)}))
        # Manually set high pest pressure in state (simulate outbreak)
        state = env.state()
        for slot in state.crop_slots:
            slot.pest_pressure = 0.9
        action = agent.act(state)
        # Eco mode should use biological (1), not chemical (2)
        for v in action.pest_control.values():
            assert v != 2, "Eco mode agent should not use chemical pesticides"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
