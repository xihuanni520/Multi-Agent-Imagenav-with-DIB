import attr
import gzip
import json
import os
from typing import List, Optional, Iterable, Dict, Any

from habitat.config import Config, read_write
from habitat.core.utils import not_none_validator
from habitat.core.dataset import ALL_SCENES_MASK, Dataset, Episode, BaseEpisode
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
from habitat.datasets.pointnav.pointnav_dataset import (
    PointNavDatasetV1, 
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
) 


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoalV2(NavigationGoal):
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None
    views: Optional[List[float]] = None


@attr.s(auto_attribs=True, kw_only=True)
class AgentSpec:
    agent_id: int = attr.ib(validator=not_none_validator)
    start_position: List[float] = attr.ib(default=None, validator=not_none_validator)
    start_rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    goals: List[NavigationGoalV2] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[dict] = None


@attr.s(auto_attribs=True, kw_only=True)
class MultiAgentNavigationEpisode(BaseEpisode):
    agents: List[AgentSpec] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[dict] = None


@registry.register_dataset(name="ImageNav-v1")
class ImageNavDatasetV1(PointNavDatasetV1):
    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoalV2(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)


@registry.register_dataset(name="MultiImageNav-v1")
class MultiImageNavDatasetV1(Dataset):
    r"""Dataset for multi-agent ImageNav episodes.

    Expected JSON format:
    {
      "episodes": [
        {
          "scene_id": "...",
          "episode_id": "...",
          "agents": [
            {"agent_id": 0, "start_position": [...], "start_rotation": [...],
             "goals": [{"position": [...], "radius": null}], "info": {...}},
            ...
          ],
          "info": {...}
        },
        ...
      ]
    }
    """

    episodes: List[MultiAgentNavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        data_path = config.data_path
        if isinstance(data_path, str) and "{split}" in data_path:
            data_path = data_path.format(split=config.split)
        return os.path.exists(data_path)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Stream dataset to collect scene ids without loading all episodes."""
        scenes = list(getattr(config, "content_scenes", []))
        if len(scenes) > 0 and ALL_SCENES_MASK not in scenes:
            return scenes

        data_path = config.data_path
        if isinstance(data_path, str) and "{split}" in data_path:
            data_path = data_path.format(split=config.split)
        scenes_dir = getattr(config, "scenes_dir", None)
        scene_set = set()
        with gzip.open(data_path, "rt") as f:
            for episode_data in cls._iter_episode_dicts(f):
                scene_id = episode_data.get("scene_id", "")
                if not scene_id:
                    continue
                scene_path = scene_id
                if scenes_dir is not None:
                    if scene_path.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        scene_path = scene_path[len(DEFAULT_SCENE_PATH_PREFIX) :]
                    if not os.path.isabs(scene_path):
                        scene_path = os.path.join(scenes_dir, scene_path)
                    if not os.path.exists(scene_path):
                        continue
                scene_set.add(cls.scene_from_scene_path(scene_id))
        scenes = sorted(scene_set)
        if len(scenes) == 0:
            return [ALL_SCENES_MASK]
        return scenes

    @staticmethod
    def _iter_episode_dicts(fp) -> Iterable[Dict[str, Any]]:
        r"""Stream JSON episodes without loading full file into memory."""
        decoder = json.JSONDecoder()
        buf = ""
        in_array = False

        while True:
            chunk = fp.read(65536)
            if not chunk:
                break
            buf += chunk

            if not in_array:
                idx = buf.find('"episodes"')
                if idx == -1:
                    continue
                idx = buf.find("[", idx)
                if idx == -1:
                    continue
                buf = buf[idx + 1 :]
                in_array = True

            while True:
                buf = buf.lstrip()
                if buf.startswith("]"):
                    return
                try:
                    obj, offset = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break
                yield obj
                buf = buf[offset:]
                buf = buf.lstrip()
                if buf.startswith(","):
                    buf = buf[1:]

    def _build_episode(
        self, episode_data: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> MultiAgentNavigationEpisode:
        if scenes_dir is not None:
            if episode_data["scene_id"].startswith(DEFAULT_SCENE_PATH_PREFIX):
                episode_data["scene_id"] = episode_data["scene_id"][
                    len(DEFAULT_SCENE_PATH_PREFIX) :
                ]
            if not os.path.isabs(episode_data["scene_id"]):
                episode_data["scene_id"] = os.path.join(
                    scenes_dir, episode_data["scene_id"]
                )

        agents = []
        for agent_data in episode_data["agents"]:
            goals = [
                NavigationGoalV2(**g) for g in agent_data.get("goals", [])
            ]
            agent = AgentSpec(
                agent_id=agent_data["agent_id"],
                start_position=agent_data["start_position"],
                start_rotation=agent_data["start_rotation"],
                goals=goals,
                info=agent_data.get("info", None),
            )
            agents.append(agent)

        return MultiAgentNavigationEpisode(
            episode_id=episode_data["episode_id"],
            scene_id=episode_data["scene_id"],
            agents=agents,
            info=episode_data.get("info", None),
        )

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        if config is None:
            return
        datasetfile_path = config.data_path
        if isinstance(datasetfile_path, str) and "{split}" in datasetfile_path:
            datasetfile_path = datasetfile_path.format(split=config.split)
        scenes_dir = getattr(config, "scenes_dir", None)
        content_scenes = list(getattr(config, "content_scenes", []))
        filter_scenes = (
            len(content_scenes) > 0 and ALL_SCENES_MASK not in content_scenes
        )

        with gzip.open(datasetfile_path, "rt") as f:
            for episode_data in self._iter_episode_dicts(f):
                episode = self._build_episode(episode_data, scenes_dir)
                if filter_scenes:
                    scene_name = self.scene_from_scene_path(episode.scene_id)
                    if scene_name not in content_scenes:
                        continue
                if scenes_dir is not None and not os.path.exists(episode.scene_id):
                    # Skip episodes whose scene asset is missing locally
                    continue
                self.episodes.append(episode)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode_data in deserialized["episodes"]:
            episode = self._build_episode(episode_data, scenes_dir)
            self.episodes.append(episode)
