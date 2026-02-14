from typing import Optional, Dict, Any, List

import gym
import habitat
import numpy as np
from habitat import Config, Dataset, ThreadedVectorEnv
from habitat.utils.gym_adapter import create_action_space
from habitat.config import read_write
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.gym_adapter import HabGymWrapper

from src.dataset import MultiImageNavDatasetV1, MultiAgentNavigationEpisode, NavigationGoalV2
from habitat.tasks.nav.nav import NavigationEpisode


class NavRLEnvX(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        # self._core_env_config = config.TASK_CONFIG
        if len(self._config.task.reward_measure) > 0:
            self._reward_measure_names = [self._config.task.reward_measure]
            self._reward_scales = [1.0]
        else:
            self._reward_measure_names = self._config.task.reward_measure
            self._reward_scales = self._config.task.reward_scales

        self._success_measure_name = self._config.task.success_measure
        habitat.logger.info('NavRLEnvX: '
                            f'Reward Measures={self._reward_measure_names}, '
                            f'Reward Scales={self._reward_scales}, '
                            f'Success Measure={self._success_measure_name}')
        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._config, dataset)
        # Convert habitat ActionSpace to gym space for baselines utilities.
        self._original_action_space = self.action_space
        self.action_space = create_action_space(self._original_action_space)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._get_reward_measure()
        return observations

    def _get_reward_measure(self):
        current_measure = 0.0
        for reward_measure_name, reward_scale in zip(
                self._reward_measure_names, self._reward_scales
        ):
            if "." in reward_measure_name:
                reward_measure_name = reward_measure_name.split('.')
                measure = self._env.get_metrics()[
                    reward_measure_name[0]
                ][reward_measure_name[1]]
            else:
                measure = self._env.get_metrics()[reward_measure_name]
            current_measure += measure * reward_scale
        return current_measure

    def step(self, *args, **kwargs):
        if "action" in kwargs:
            self._previous_action = kwargs["action"]
        elif len(args) > 0:
            self._previous_action = args[0]
        else:
            self._previous_action = None
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._config.task.slack_reward - 1.0,
            self._config.task.success_reward + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config.task.slack_reward

        current_measure = self._get_reward_measure()

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._episode_success() * self._config.task.success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def set_current_episode(self, episode: NavigationEpisode) -> None:
        self._env.current_episode = episode

    @property
    def original_action_space(self):
        # Match HabGymWrapper attribute expected by VectorEnv
        return getattr(self, "_original_action_space", self.action_space)


class MultiNavRLEnvX(NavRLEnvX):
    """Single-agent env used inside the multi-agent wrapper.

    Ignore success for done; only terminate on episode_over (max steps).
    """

    def get_done(self, observations):
        return self._env.episode_over


@habitat.registry.register_env(name="GymHabitatEnvX")
class GymHabitatEnvX(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        base_env = NavRLEnvX(config=config, dataset=dataset)
        env = HabGymWrapper(base_env)
        super().__init__(env)


@habitat.registry.register_env(name="MultiGymHabitatEnvX")
class MultiGymHabitatEnvX(gym.Env):
    """Multi-agent wrapper: each env hosts N single-agent Habitat envs.

    This keeps action/obs spaces compatible with existing trainers by
    stacking observations in the agent dimension and flattening in trainer.
    """

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._dataset: MultiImageNavDatasetV1 = dataset  # type: ignore
        self._num_agents = int(getattr(config.task, "num_agents", 1))
        self._parallel_sub_envs = bool(
            getattr(config.task, "parallel_sub_envs", False)
        )
        if self._num_agents < 1:
            raise ValueError("num_agents must be >= 1")

        # Multi-agent sync requires continuing after per-agent success.
        # Force end_on_success off to avoid Habitat Env asserting on step().
        with read_write(self._config):
            self._config.task.end_on_success = False

        self._max_episode_steps = config.environment.max_episode_steps
        self._elapsed_steps = 0

        self._episode_iterator = None
        if self._dataset is not None:
            self._episode_iterator = self._dataset.get_episode_iterator(
                **config.environment.iterator_options
            )
        self._current_episode: Optional[MultiAgentNavigationEpisode] = None
        self._episode_from_iter_on_reset = True

        # Create sub-envs with a dummy dataset; current_episode is set on reset.
        dummy_episode = NavigationEpisode(
            episode_id="dummy",
            scene_id=config.simulator.scene,
            start_position=[0.0, 0.0, 0.0],
            start_rotation=[0.0, 0.0, 0.0, 1.0],
            goals=[NavigationGoalV2(position=[0.0, 0.0, 0.0], radius=1.0)],
        )
        if self._parallel_sub_envs:
            def _make_sub_env(rank: int = 0):
                dummy_dataset = habitat.make_dataset("ImageNav-v1")
                dummy_dataset.episodes = [dummy_episode]
                base_env = MultiNavRLEnvX(config=config, dataset=dummy_dataset)
                base_env.seed(config.seed + rank)
                return base_env

            self._sub_envs = ThreadedVectorEnv(
                make_env_fn=_make_sub_env,
                env_fn_args=tuple((i,) for i in range(self._num_agents)),
                auto_reset_done=False,
            )
            # Expose single-agent action/obs spaces to keep trainer compatibility.
            self.observation_space = self._sub_envs.observation_spaces[0]
            self.action_space = self._sub_envs.action_spaces[0]
            self.original_action_space = self._sub_envs.orig_action_spaces[0]
        else:
            self._sub_envs: List[gym.Env] = []
            for _ in range(self._num_agents):
                dummy_dataset = habitat.make_dataset("ImageNav-v1")
                dummy_dataset.episodes = [dummy_episode]
                base_env = MultiNavRLEnvX(config=config, dataset=dummy_dataset)
                self._sub_envs.append(HabGymWrapper(base_env))

            # Expose single-agent action/obs spaces to keep trainer compatibility.
            self.observation_space = self._sub_envs[0].observation_space
            self.action_space = self._sub_envs[0].action_space
            self.original_action_space = getattr(
                self._sub_envs[0], "original_action_space", self.action_space
            )

        self._agent_done = np.zeros((self._num_agents,), dtype=bool)
        self._agent_last_obs: List[Optional[Dict[str, Any]]] = [
            None for _ in range(self._num_agents)
        ]
        self._agent_last_info: List[Dict[str, Any]] = [
            {} for _ in range(self._num_agents)
        ]

    def _make_agent_episode(
        self, episode: MultiAgentNavigationEpisode, agent_idx: int
    ) -> NavigationEpisode:
        agent_spec = episode.agents[agent_idx]
        return NavigationEpisode(
            episode_id=f"{episode.episode_id}_agent{agent_spec.agent_id}",
            scene_id=episode.scene_id,
            start_position=agent_spec.start_position,
            start_rotation=agent_spec.start_rotation,
            goals=[NavigationGoalV2(**g.__dict__) for g in agent_spec.goals],
        )

    def _get_next_episode(self) -> MultiAgentNavigationEpisode:
        if self._episode_iterator is None:
            raise RuntimeError("Dataset is required for MultiGymHabitatEnvX")
        return next(self._episode_iterator)

    def reset(self):
        if self._episode_from_iter_on_reset or self._current_episode is None:
            self._current_episode = self._get_next_episode()
        self._episode_from_iter_on_reset = True
        self._elapsed_steps = 0
        self._agent_done[:] = False
        self._agent_last_obs = [None for _ in range(self._num_agents)]
        self._agent_last_info = [{} for _ in range(self._num_agents)]

        obs_list: List[Dict[str, Any]] = []
        if self._parallel_sub_envs:
            for i in range(self._num_agents):
                agent_episode = self._make_agent_episode(self._current_episode, i)
                self._sub_envs.call_at(
                    i, "set_current_episode", {"episode": agent_episode}
                )
            for i in range(self._num_agents):
                obs = self._sub_envs.reset_at(i)[0]
                self._agent_last_obs[i] = obs
                self._agent_last_info[i] = {}
                obs_list.append(obs)
        else:
            for i, env in enumerate(self._sub_envs):
                agent_episode = self._make_agent_episode(self._current_episode, i)
                # Set episode manually before reset to ensure proper scene/goal.
                env._env.set_current_episode(agent_episode)  # type: ignore[attr-defined]
                obs = env.reset()
                self._agent_last_obs[i] = obs
                self._agent_last_info[i] = {}
                obs_list.append(obs)

        return self._stack_observations(obs_list)

    def step(self, actions):
        if self._current_episode is None:
            raise RuntimeError("Call reset() before step().")

        if isinstance(actions, (np.ndarray, list)):
            if len(actions) != self._num_agents:
                raise ValueError(
                    f"Expected {self._num_agents} actions, got {len(actions)}"
                )
            action_list = list(actions)
        else:
            # allow scalar broadcast
            action_list = [actions for _ in range(self._num_agents)]

        obs_list: List[Dict[str, Any]] = []
        reward_list: List[float] = []
        info_list: List[Dict[str, Any]] = []
        done_list: List[bool] = []

        if self._parallel_sub_envs:
            # async step for active agents
            for i in range(self._num_agents):
                if not self._agent_done[i]:
                    self._sub_envs.async_step_at(i, action_list[i])
            for i in range(self._num_agents):
                if self._agent_done[i]:
                    obs = self._agent_last_obs[i]
                    if obs is None:
                        obs = self._sub_envs.reset_at(i)[0]
                        self._agent_last_obs[i] = obs
                    reward = 0.0
                    info = self._agent_last_info[i]
                    done = False
                else:
                    obs, reward, done, info = self._sub_envs.wait_step_at(i)
                    self._agent_last_obs[i] = obs
                    self._agent_last_info[i] = info
                obs_list.append(obs)
                reward_list.append(float(reward))
                info_list.append(info)
                done_list.append(bool(done))
        else:
            for i, env in enumerate(self._sub_envs):
                if self._agent_done[i]:
                    # keep agent stationary without advancing its episode
                    obs = self._agent_last_obs[i]
                    if obs is None:
                        obs = env.reset()
                        self._agent_last_obs[i] = obs
                    reward = 0.0
                    info = self._agent_last_info[i]
                    done = False
                else:
                    obs, reward, done, info = env.step(action_list[i])
                    self._agent_last_obs[i] = obs
                    self._agent_last_info[i] = info
                obs_list.append(obs)
                reward_list.append(float(reward))
                info_list.append(info)
                done_list.append(bool(done))

        # Compute per-agent success from info if available.
        agent_success = np.array(
            [info.get("success", 0.0) for info in info_list], dtype=np.float32
        )
        # Mark done when success or when sub-env ended.
        agent_episode_over = np.array(done_list, dtype=bool)
        self._agent_done = np.logical_or(
            self._agent_done, np.logical_or(agent_success > 0.0, agent_episode_over)
        )

        self._elapsed_steps += 1
        done_all = bool(self._agent_done.all())
        if self._elapsed_steps >= self._max_episode_steps:
            done_all = True
            self._agent_done[:] = True

        stacked_obs = self._stack_observations(obs_list)
        info = {
            "agent_done": self._agent_done.copy().tolist(),
            "agent_success": agent_success.tolist(),
        }
        # Provide per-agent infos under a namespaced key for debugging.
        info["agents_info"] = info_list

        # Multi-agent metrics (per-agent + aggregate)
        def _metric_list(key: str, default: float = 0.0):
            out = []
            for agent_info in info_list:
                v = agent_info.get(key, default)
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float(default))
            return out

        success_list = _metric_list("success", 0.0)
        spl_list = _metric_list("spl", 0.0)
        # Prefer distance_to_goal if available; fall back to distance_to_view
        if any("distance_to_goal" in ainfo for ainfo in info_list):
            distance_list = _metric_list("distance_to_goal", 0.0)
        else:
            distance_list = _metric_list("distance_to_view", 0.0)

        ma_metrics: Dict[str, float] = {
            "success_mean": float(np.mean(success_list)),
            "success_all": float(np.all(np.array(success_list) > 0.0)),
            "spl_mean": float(np.mean(spl_list)),
            "distance_mean": float(np.mean(distance_list)),
        }
        for i in range(self._num_agents):
            ma_metrics[f"success_{i}"] = float(success_list[i])
            ma_metrics[f"spl_{i}"] = float(spl_list[i])
            ma_metrics[f"distance_{i}"] = float(distance_list[i])
        info["multi_agent"] = ma_metrics
        # Aggregate scalar metrics across agents for logging.
        agg_metrics: Dict[str, float] = {}
        for k in info_list[0].keys():
            vals = []
            for agent_info in info_list:
                v = agent_info.get(k, None)
                if v is not None and np.size(v) == 1 and not isinstance(v, str):
                    vals.append(float(v))
            if len(vals) > 0:
                agg_metrics[k] = float(np.mean(vals))
        info.update(agg_metrics)

        return stacked_obs, np.array(reward_list, dtype=np.float32), done_all, info

    def _stack_observations(self, obs_list: List[Dict[str, Any]]):
        if isinstance(obs_list[0], dict):
            out = {}
            for k in obs_list[0].keys():
                out[k] = np.stack([o[k] for o in obs_list], axis=0)
            return out
        else:
            return np.stack(obs_list, axis=0)

    def close(self):
        if self._parallel_sub_envs:
            self._sub_envs.close()
        else:
            for env in self._sub_envs:
                env.close()

    def seed(self, seed: Optional[int] = None):
        if self._parallel_sub_envs:
            for i in range(self._num_agents):
                self._sub_envs.call_at(i, "seed", {"seed": seed})
        else:
            for env in self._sub_envs:
                env.seed(seed)

    @property
    def number_of_episodes(self) -> int:
        if self._dataset is None:
            return 0
        return len(self._dataset.episodes)

    def current_episode(self, all_info: bool = False):
        if self._current_episode is None:
            return None
        if all_info:
            return self._current_episode
        # Minimal episode info for compatibility
        return type(
            "EpisodeInfo",
            (),
            {
                "episode_id": self._current_episode.episode_id,
                "scene_id": self._current_episode.scene_id,
            },
        )()

    @property
    def episodes(self):
        if self._dataset is None:
            return []
        return self._dataset.episodes
