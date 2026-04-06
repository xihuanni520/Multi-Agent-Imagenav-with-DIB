from typing import Optional, Dict, Any, List, Tuple
import signal
import traceback
import random
import gzip
import hashlib
import json
import os
import tempfile
import time

import gym
import habitat
import numpy as np
from habitat import Config, Dataset, ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat.config import read_write
from habitat.utils.gym_adapter import HabGymWrapper, create_action_space, flatten_dict
from torch import multiprocessing as mp

from src.dataset import (
    ImageNavDatasetV1,
    MultiImageNavDatasetV1,
    MultiAgentNavigationEpisode,
    NavigationGoalV2,
)
from habitat.tasks.nav.nav import NavigationEpisode


class NavRLEnvX(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        if len(self._config.task.reward_measure) > 0:
            self._reward_measure_names = [self._config.task.reward_measure]
            self._reward_scales = [1.0]
        else:
            self._reward_measure_names = self._config.task.reward_measure
            self._reward_scales = self._config.task.reward_scales

        self._success_measure_name = self._config.task.success_measure
        habitat.logger.info(
            "NavRLEnvX: "
            f"Reward Measures={self._reward_measure_names}, "
            f"Reward Scales={self._reward_scales}, "
            f"Success Measure={self._success_measure_name}"
        )
        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._config, dataset)
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
                reward_measure_name = reward_measure_name.split(".")
                measure = self._env.get_metrics()[reward_measure_name[0]][
                    reward_measure_name[1]
                ]
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
        return getattr(self, "_original_action_space", self.action_space)


class MultiNavRLEnvX(NavRLEnvX):
    def get_done(self, observations):
        return self._env.episode_over


@habitat.registry.register_env(name="GymHabitatEnvX")
class GymHabitatEnvX(gym.Wrapper):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        base_env = NavRLEnvX(config=config, dataset=dataset)
        env = HabGymWrapper(base_env)
        super().__init__(env)


class MultiAgentProcessGymWrapper(gym.Wrapper):
    def __init__(self, env: MultiNavRLEnvX):
        super().__init__(env)
        self.action_space = env.action_space
        self.original_action_space = env.original_action_space
        self.observation_space = env.observation_space
        self._last_obs = None

    def _transform_obs(self, obs):
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    out[k] = np.array(v, copy=True)
                else:
                    out[k] = v
            return out
        return obs

    def reset(self):
        obs = self.env.reset()
        self._last_obs = obs
        return self._transform_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        return self._transform_obs(obs), reward, done, flatten_dict(info)


class MultiAgentProcessVectorEnv:
    def __init__(
        self,
        env_fn_args: List[Tuple[Config, "MultiAgentProcessGroupSync", int, int]],
        auto_reset_done: bool = True,
        workers_ignore_signals: bool = False,
        mp_ctx=None,
    ):
        self._mp_ctx = mp_ctx or mp.get_context("spawn")
        self._auto_reset_done = auto_reset_done
        self._workers_ignore_signals = workers_ignore_signals
        self._parent_conns = []
        self._workers = []
        self._closed = False

        for env_args in env_fn_args:
            parent_conn, child_conn = self._mp_ctx.Pipe(duplex=True)
            proc = self._mp_ctx.Process(
                target=_multi_agent_process_worker,
                args=(
                    child_conn,
                    env_args,
                    auto_reset_done,
                    workers_ignore_signals,
                ),
            )
            proc.daemon = False
            proc.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._workers.append(proc)

        self.observation_spaces = []
        self.action_spaces = []
        self.orig_action_spaces = []
        self.number_of_episodes = []
        for conn in self._parent_conns:
            cmd, payload = conn.recv()
            if cmd != "spaces":
                raise RuntimeError(f"Unexpected worker init message: {cmd}")
            if len(payload) == 4:
                obs_space, action_space, orig_action_space, num_eps = payload
            else:
                obs_space, action_space, orig_action_space = payload
                num_eps = -1
            self.observation_spaces.append(obs_space)
            self.action_spaces.append(action_space)
            self.orig_action_spaces.append(orig_action_space)
            self.number_of_episodes.append(int(num_eps))

    @property
    def num_envs(self):
        return len(self._parent_conns)

    def reset(self):
        for conn in self._parent_conns:
            conn.send(("reset", None))
        return [conn.recv() for conn in self._parent_conns]

    def reset_at(self, index_env: int):
        self._parent_conns[index_env].send(("reset", None))
        return [self._parent_conns[index_env].recv()]

    def async_step_at(self, index_env: int, action):
        self._parent_conns[index_env].send(("step", action))

    def wait_step_at(self, index_env: int):
        return self._parent_conns[index_env].recv()

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {len(actions)}"
            )
        for idx, action in enumerate(actions):
            self.async_step_at(idx, action)
        return [self.wait_step_at(idx) for idx in range(self.num_envs)]

    def current_episodes(self):
        for conn in self._parent_conns:
            conn.send(("current_episode", None))
        return [conn.recv() for conn in self._parent_conns]

    def pause_at(self, index_env: int) -> None:
        conn = self._parent_conns.pop(index_env)
        worker = self._workers.pop(index_env)
        self.observation_spaces.pop(index_env)
        self.action_spaces.pop(index_env)
        self.orig_action_spaces.pop(index_env)
        self.number_of_episodes.pop(index_env)
        try:
            conn.send(("close", None))
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        worker.join(timeout=5.0)
        if worker.is_alive():
            worker.terminate()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for conn in self._parent_conns:
            try:
                conn.close()
            except Exception:
                pass
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()

    def __del__(self):
        self.close()


class MultiAgentProcessGroupSync:
    def __init__(self, num_agents: int, max_episode_steps: int, mp_ctx=None):
        ctx = mp_ctx or mp.get_context("spawn")
        self.num_agents = num_agents
        self.max_episode_steps = max_episode_steps
        self.condition = ctx.Condition()
        self.step_phase = ctx.Value("i", 0)
        self.step_waiting = ctx.Value("i", 0)
        self.reset_phase = ctx.Value("i", 0)
        self.reset_waiting = ctx.Value("i", 0)
        self.group_step = ctx.Value("i", 0)
        self.group_done = ctx.Value("b", False)
        self.done_flags = ctx.Array("b", [False] * num_agents)
        self.success_flags = ctx.Array("d", [0.0] * num_agents)
        self.spl_values = ctx.Array("d", [0.0] * num_agents)
        self.distance_values = ctx.Array("d", [0.0] * num_agents)

    def on_step(
        self,
        agent_idx: int,
        local_done: bool,
        local_success: float,
        spl: float,
        distance: float,
    ) -> Tuple[bool, List[bool], List[float], List[float], List[float], int]:
        with self.condition:
            current_phase = self.step_phase.value
            if local_done:
                self.done_flags[agent_idx] = True
            self.success_flags[agent_idx] = float(local_success)
            self.spl_values[agent_idx] = float(spl)
            self.distance_values[agent_idx] = float(distance)
            self.step_waiting.value += 1
            if self.step_waiting.value == self.num_agents:
                self.group_step.value += 1
                group_done = all(bool(v) for v in self.done_flags[:]) or (
                    self.group_step.value >= self.max_episode_steps
                )
                self.group_done.value = bool(group_done)
                self.step_waiting.value = 0
                self.step_phase.value += 1
                self.condition.notify_all()
            else:
                while current_phase == self.step_phase.value:
                    self.condition.wait()

            return (
                bool(self.group_done.value),
                [bool(v) for v in self.done_flags[:]],
                list(self.success_flags[:]),
                list(self.spl_values[:]),
                list(self.distance_values[:]),
                int(self.group_step.value),
            )

    def on_reset(self) -> None:
        with self.condition:
            current_phase = self.reset_phase.value
            self.reset_waiting.value += 1
            if self.reset_waiting.value == self.num_agents:
                self.group_step.value = 0
                self.group_done.value = False
                for idx in range(self.num_agents):
                    self.done_flags[idx] = False
                    self.success_flags[idx] = 0.0
                    self.spl_values[idx] = 0.0
                    self.distance_values[idx] = 0.0
                self.reset_waiting.value = 0
                self.reset_phase.value += 1
                self.condition.notify_all()
            else:
                while current_phase == self.reset_phase.value:
                    self.condition.wait()


def _build_single_agent_dataset(
    dataset: MultiImageNavDatasetV1, agent_idx: int
) -> ImageNavDatasetV1:
    agent_dataset = ImageNavDatasetV1()
    agent_dataset.episodes = []
    for episode in dataset.episodes:
        agent_spec = episode.agents[agent_idx]
        agent_dataset.episodes.append(
            NavigationEpisode(
                episode_id=f"{episode.episode_id}_agent{agent_spec.agent_id}",
                scene_id=episode.scene_id,
                start_position=agent_spec.start_position,
                start_rotation=agent_spec.start_rotation,
                goals=[NavigationGoalV2(**g.__dict__) for g in agent_spec.goals],
            )
        )
    return agent_dataset


class MultiAgentProcessEnvX(MultiNavRLEnvX):
    def __init__(
        self,
        config: Config,
        dataset: Optional[Dataset] = None,
        group_sync: Optional[MultiAgentProcessGroupSync] = None,
        agent_index: int = 0,
        group_index: int = 0,
    ):
        self._group_sync = group_sync
        self._agent_index = agent_index
        self._group_index = group_index
        self._local_done = False
        self._last_obs = None
        self._last_info: Dict[str, Any] = {}
        self._last_reward = 0.0

        if isinstance(dataset, MultiImageNavDatasetV1):
            agent_dataset = _build_single_agent_dataset(dataset, agent_index)
        elif isinstance(dataset, ImageNavDatasetV1):
            agent_dataset = dataset
        else:
            raise TypeError(
                "MultiAgentProcessEnvX expects MultiImageNavDatasetV1 or "
                f"ImageNavDatasetV1, got {type(dataset)}"
            )
        with read_write(config):
            config.task.end_on_success = False
            config.task.parallel_sub_envs = False
            config.task.multi_agent_process_mode = True
        super().__init__(config=config, dataset=agent_dataset)

    def _get_episode_metadata(self) -> Dict[str, str]:
        current_episode = getattr(self.habitat_env, "current_episode", None)
        scene_id = getattr(current_episode, "scene_id", "")
        episode_id = getattr(current_episode, "episode_id", "")
        if isinstance(episode_id, str) and "_agent" in episode_id:
            group_episode_id = episode_id.rsplit("_agent", 1)[0]
        else:
            group_episode_id = episode_id
        return {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "group_episode_id": group_episode_id,
        }

    def reset(self):
        if self._group_sync is not None:
            self._group_sync.on_reset()
        obs = super().reset()
        self._local_done = False
        self._last_obs = obs
        self._last_info = self.get_info(obs)
        self._last_reward = 0.0
        return obs

    def step(self, action):
        if self._local_done:
            obs = self._last_obs
            reward = 0.0
            info = dict(self._last_info)
            local_done = True
        else:
            obs, reward, _, info = super().step(action)
            local_success = float(self._episode_success())
            local_done = bool(local_success > 0.0 or self.habitat_env.episode_over)
            if local_done:
                self._local_done = True
            self._last_obs = obs
            self._last_info = dict(info)
            self._last_reward = float(reward)

        info = dict(self._last_info)
        local_success = float(info.get("success", 0.0))
        local_spl = float(info.get("spl", 0.0))
        if "distance_to_goal" in info:
            local_distance = float(info.get("distance_to_goal", 0.0))
        else:
            local_distance = float(info.get("distance_to_view", 0.0))
        episode_meta = self._get_episode_metadata()

        if self._group_sync is None:
            info.update(
                {
                    "group_id": self._group_index,
                    "agent_id": self._agent_index,
                    "agent_done": float(local_done),
                    **episode_meta,
                }
            )
            return self._last_obs, self._last_reward, local_done, info

        (
            group_done,
            done_flags,
            success_flags,
            spl_values,
            distance_values,
            group_step,
        ) = self._group_sync.on_step(
            self._agent_index,
            local_done,
            local_success,
            local_spl,
            local_distance,
        )

        multi_agent_metrics = {
            "success_mean": float(np.mean(success_flags)),
            "success_all": float(np.all(np.array(success_flags) > 0.0)),
            "spl_mean": float(np.mean(spl_values)),
            "distance_mean": float(np.mean(distance_values)),
        }
        for idx in range(len(success_flags)):
            multi_agent_metrics[f"success_{idx}"] = float(success_flags[idx])
            multi_agent_metrics[f"spl_{idx}"] = float(spl_values[idx])
            multi_agent_metrics[f"distance_{idx}"] = float(distance_values[idx])

        info.update(
            {
                "group_id": self._group_index,
                "agent_id": self._agent_index,
                "agent_done": float(done_flags[self._agent_index]),
                "group_done": float(group_done),
                "group_step": float(group_step),
                "multi_agent": multi_agent_metrics,
                **episode_meta,
            }
        )

        return self._last_obs, self._last_reward, group_done, info


@habitat.registry.register_env(name="MultiGymHabitatEnvX")
class MultiGymHabitatEnvX(gym.Env):
    """Legacy wrapper: one process hosts N agents via stacked observations."""

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._dataset: MultiImageNavDatasetV1 = dataset  # type: ignore
        self._num_agents = int(getattr(config.task, "num_agents", 1))
        self._parallel_sub_envs = bool(
            getattr(config.task, "parallel_sub_envs", False)
        )
        if self._num_agents < 1:
            raise ValueError("num_agents must be >= 1")

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
            action_list = [actions for _ in range(self._num_agents)]

        obs_list: List[Dict[str, Any]] = []
        reward_list: List[float] = []
        info_list: List[Dict[str, Any]] = []
        done_list: List[bool] = []

        if self._parallel_sub_envs:
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

        agent_success = np.array(
            [info.get("success", 0.0) for info in info_list], dtype=np.float32
        )
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
            "scene_id": self._current_episode.scene_id,
            "episode_id": self._current_episode.episode_id,
            "group_episode_id": self._current_episode.episode_id,
        }
        info["agents_info"] = info_list

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


def _make_multi_agent_process_env_fn(
    config: Config,
    group_sync: MultiAgentProcessGroupSync,
    agent_index: int,
    group_index: int,
) -> gym.Env:
    task_config = config.habitat if "habitat" in config else config
    dataset = make_dataset(task_config.dataset.type, config=task_config.dataset)
    base_env = MultiAgentProcessEnvX(
        config=task_config,
        dataset=dataset,
        group_sync=group_sync,
        agent_index=agent_index,
        group_index=group_index,
    )
    base_env.seed(task_config.seed)
    return MultiAgentProcessGymWrapper(base_env)


def _multi_agent_process_worker(
    child_conn,
    env_fn_args: Tuple[Config, MultiAgentProcessGroupSync, int, int],
    auto_reset_done: bool,
    workers_ignore_signals: bool,
) -> None:
    if workers_ignore_signals:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGUSR2, signal.SIG_IGN)

    env = None
    try:
        env = _make_multi_agent_process_env_fn(*env_fn_args)
        pending_reset = env.reset()
        num_eps = getattr(env, "number_of_episodes", -1)
        if callable(num_eps):
            num_eps = num_eps()
        try:
            num_eps = int(num_eps)
        except Exception:
            num_eps = -1
        child_conn.send(
            (
                "spaces",
                (
                    env.observation_space,
                    env.action_space,
                    getattr(env, "original_action_space", env.action_space),
                    num_eps,
                ),
            )
        )

        while True:
            cmd, payload = child_conn.recv()
            if cmd == "reset":
                if pending_reset is not None:
                    obs = pending_reset
                    pending_reset = None
                else:
                    obs = env.reset()
                child_conn.send(obs)
            elif cmd == "step":
                obs, reward, done, info = env.step(payload)
                if auto_reset_done and done:
                    obs = env.reset()
                child_conn.send((obs, reward, done, info))
            elif cmd == "current_episode":
                current_episode = getattr(env, "current_episode", None)
                if callable(current_episode):
                    current_episode = current_episode()
                child_conn.send(current_episode)
            elif cmd == "close":
                break
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except Exception:
        logger.error(
            "Multi-agent process worker crashed:\n%s",
            traceback.format_exc(),
        )
        raise
    finally:
        try:
            child_conn.close()
        except Exception:
            pass
        if env is not None:
            env.close()


def construct_multi_agent_process_envs(
    config: Config,
    workers_ignore_signals: bool = False,
    enforce_scenes_greater_eq_environments: bool = False,
    world_rank: int = 0,
    world_size: int = 1,
) -> MultiAgentProcessVectorEnv:
    mp_ctx = mp.get_context("spawn")
    num_groups = int(config.habitat_baselines.num_environments)
    num_agents = int(getattr(config.habitat.task, "num_agents", 1))
    scenes = config.habitat.dataset.content_scenes
    if "*" in config.habitat.dataset.content_scenes:
        if config.habitat.dataset.type == "MultiImageNav-v1":
            scenes = MultiImageNavDatasetV1.get_scenes_to_load(
                config.habitat.dataset
            )
        else:
            dataset = make_dataset(
                config.habitat.dataset.type, config=config.habitat.dataset
            )
            scenes = dataset.get_scenes_to_load(config.habitat.dataset)
    logger.info("multi-agent dataset scene pool size(before sharding): %d", len(scenes))

    # Deterministic global shuffle to avoid alphabetical shards and ensure
    # disjoint rank-level scene pools when world_size > 1.
    scenes = list(scenes)
    base_seed = int(config.habitat.seed) - int(world_rank) * int(
        config.habitat_baselines.num_environments
    )
    rng = random.Random(base_seed)
    rng.shuffle(scenes)

    if world_size > 1:
        if len(scenes) < world_size:
            logger.warn(
                "world_size (%d) > number of scenes (%d). "
                "Each rank will use the same scene pool.",
                world_size,
                len(scenes),
            )
        else:
            scenes = scenes[world_rank::world_size]
            logger.info(
                "rank-level scene sharding: rank=%d/%d, assigned_scenes=%d",
                world_rank,
                world_size,
                len(scenes),
            )
    else:
        logger.info(
            "single-rank run: using full scene pool, assigned_scenes=%d",
            len(scenes),
        )

    if num_groups < 1:
        raise RuntimeError("num_environments must be strictly positive")
    if len(scenes) == 0:
        raise RuntimeError(
            "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
        )

    scene_splits: List[List[str]] = [[] for _ in range(num_groups)]
    if len(scenes) < num_groups:
        msg = f"There are less scenes ({len(scenes)}) than environments ({num_groups}). "
        if enforce_scenes_greater_eq_environments:
            logger.warn(msg + "Reducing the number of environments to be the number of scenes.")
            num_groups = len(scenes)
            scene_splits = [[s] for s in scenes]
        else:
            logger.warn(msg + "Each environment group will use all the scenes instead of a subset.")
            for scene in scenes:
                for split in scene_splits:
                    split.append(scene)
    else:
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
    logger.info(
        "group scene split sizes: %s",
        [len(s) for s in scene_splits],
    )

    env_fn_args = []
    group_sync_states = [
        MultiAgentProcessGroupSync(
            num_agents=num_agents,
            max_episode_steps=config.habitat.environment.max_episode_steps,
            mp_ctx=mp_ctx,
        )
        for _ in range(num_groups)
    ]

    single_agent_cache_paths = _prepare_single_agent_cache_files(
        config=config,
        scene_splits=scene_splits,
        num_agents=num_agents,
        world_rank=world_rank,
    )

    for group_idx in range(num_groups):
        for agent_idx in range(num_agents):
            proc_config = config.clone()
            with read_write(proc_config):
                task_config = proc_config.habitat
                task_config.seed = task_config.seed + group_idx
                task_config.dataset.type = "ImageNav-v1"
                task_config.dataset.data_path = single_agent_cache_paths[
                    (group_idx, agent_idx)
                ]
                task_config.dataset.content_scenes = ["*"]
                task_config.simulator.habitat_sim_v0.gpu_device_id = (
                    config.habitat_baselines.simulator_gpu_id
                )
                task_config.simulator.agent_0.sensors = (
                    config.habitat_baselines.sensors
                )
                task_config.task.multi_agent_process_mode = True
                task_config.task.agent_index = agent_idx
                task_config.task.group_index = group_idx
            env_fn_args.append(
                (
                    proc_config,
                    group_sync_states[group_idx],
                    agent_idx,
                    group_idx,
                )
            )

    envs = MultiAgentProcessVectorEnv(
        env_fn_args=env_fn_args,
        auto_reset_done=True,
        workers_ignore_signals=workers_ignore_signals,
        mp_ctx=mp_ctx,
    )
    envs.rank_scene_pool = list(scenes)
    envs.rank_scene_pool_size = len(scenes)
    envs.world_rank = int(world_rank)
    envs.world_size = int(world_size)
    return envs


def _prepare_single_agent_cache_files(
    config: Config,
    scene_splits: List[List[str]],
    num_agents: int,
    world_rank: int,
) -> Dict[Tuple[int, int], str]:
    dataset_cfg = config.habitat.dataset
    data_path = dataset_cfg.data_path
    if isinstance(data_path, str) and "{split}" in data_path:
        data_path = data_path.format(split=dataset_cfg.split)

    signature_payload = {
        "data_path": data_path,
        "mtime": os.path.getmtime(data_path),
        "size": os.path.getsize(data_path),
        "scene_splits": scene_splits,
        "num_agents": int(num_agents),
        "world_rank": int(world_rank),
    }
    signature = hashlib.md5(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = os.path.join(
        os.path.dirname(data_path), ".multi_agent_single_cache", signature
    )
    os.makedirs(cache_root, exist_ok=True)

    cache_paths: Dict[Tuple[int, int], str] = {}
    missing = False
    for group_idx in range(len(scene_splits)):
        for agent_idx in range(num_agents):
            path = os.path.join(
                cache_root,
                f"group_{group_idx:02d}_agent_{agent_idx:02d}.json.gz",
            )
            cache_paths[(group_idx, agent_idx)] = path
            if not os.path.exists(path):
                missing = True

    if not missing:
        logger.info(
            "single-agent cache hit: %s (%d files)",
            cache_root,
            len(cache_paths),
        )
        return cache_paths

    logger.info(
        "building single-agent cache at %s (groups=%d, agents=%d)",
        cache_root,
        len(scene_splits),
        num_agents,
    )
    start_t = time.time()
    _build_single_agent_cache_files(
        data_path=data_path,
        scene_splits=scene_splits,
        num_agents=num_agents,
        cache_paths=cache_paths,
    )
    logger.info(
        "finished single-agent cache build in %.2fs",
        time.time() - start_t,
    )
    return cache_paths


def _build_single_agent_cache_files(
    data_path: str,
    scene_splits: List[List[str]],
    num_agents: int,
    cache_paths: Dict[Tuple[int, int], str],
) -> None:
    scene_to_group: Dict[str, int] = {}
    for group_idx, group_scenes in enumerate(scene_splits):
        for scene in group_scenes:
            scene_to_group[scene] = group_idx

    tmp_paths: Dict[Tuple[int, int], str] = {}
    writers: Dict[Tuple[int, int], Any] = {}
    first_item: Dict[Tuple[int, int], bool] = {}

    try:
        for key, final_path in cache_paths.items():
            group_idx, agent_idx = key
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"tmp_g{group_idx}_a{agent_idx}_",
                suffix=".json.gz",
                dir=os.path.dirname(final_path),
            )
            os.close(fd)
            tmp_paths[key] = tmp_path
            fh = gzip.open(tmp_path, "wt")
            fh.write('{"episodes":[')
            writers[key] = fh
            first_item[key] = True

        processed = 0
        kept = 0
        with gzip.open(data_path, "rt") as source:
            for episode_data in MultiImageNavDatasetV1._iter_episode_dicts(source):
                processed += 1
                scene_name = MultiImageNavDatasetV1.scene_from_scene_path(
                    episode_data.get("scene_id", "")
                )
                group_idx = scene_to_group.get(scene_name, None)
                if group_idx is None:
                    continue

                agents = episode_data.get("agents", [])
                max_agents = min(num_agents, len(agents))
                if max_agents <= 0:
                    continue

                for agent_idx in range(max_agents):
                    agent_data = agents[agent_idx]
                    out_episode = {
                        "scene_id": episode_data.get("scene_id", ""),
                        "episode_id": (
                            f"{episode_data.get('episode_id', 'episode')}"
                            f"_agent{agent_data.get('agent_id', agent_idx)}"
                        ),
                        "start_position": agent_data.get("start_position", [0, 0, 0]),
                        "start_rotation": agent_data.get(
                            "start_rotation", [0, 0, 0, 1]
                        ),
                        "goals": agent_data.get("goals", []),
                        "info": agent_data.get("info", None),
                    }
                    key = (group_idx, agent_idx)
                    fh = writers[key]
                    if not first_item[key]:
                        fh.write(",")
                    fh.write(json.dumps(out_episode, separators=(",", ":")))
                    first_item[key] = False
                    kept += 1

                if processed % 50000 == 0:
                    logger.info(
                        "single-agent cache build progress: episodes=%d",
                        processed,
                    )
    except Exception:
        for fh in writers.values():
            try:
                fh.close()
            except Exception:
                pass
        for path in tmp_paths.values():
            try:
                os.remove(path)
            except Exception:
                pass
        raise
    else:
        for key, fh in writers.items():
            fh.write("]}")
            fh.close()
            os.replace(tmp_paths[key], cache_paths[key])
    logger.info(
        "single-agent cache build summary: source_episodes=%d, written_episodes=%d",
        processed,
        kept,
    )
