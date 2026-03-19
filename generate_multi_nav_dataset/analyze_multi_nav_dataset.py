#!/usr/bin/env python3

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return math.sqrt(variance)


def _iter_episodes(path: str) -> Iterator[Dict]:
    try:
        import ijson  # type: ignore

        with gzip.open(path, "rb") as f:
            for episode in ijson.items(f, "episodes.item"):
                yield episode
        return
    except ImportError:
        pass

    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    for episode in data["episodes"]:
        yield episode


def _basename(scene_id: str) -> str:
    return scene_id.rsplit("/", 1)[-1]


def _extract_agent_distances(episode: Dict) -> List[float]:
    distances = []
    for agent in episode.get("agents", []):
        info = agent.get("info", {})
        distance = info.get("geodesic_distance", None)
        if distance is None:
            continue
        distances.append(float(distance))
    return distances


def analyze_dataset(
    path: str, top_k: int, show_blocks: int
) -> None:
    scene_stats: Dict[str, RunningStats] = defaultdict(RunningStats)
    scene_agent_counts: Dict[str, RunningStats] = defaultdict(RunningStats)
    agent_count_stats = RunningStats()
    episode_mean_dist_stats = RunningStats()

    total_episodes = 0
    scene_transitions = 0
    first_blocks: List[Tuple[str, int]] = []
    longest_block: Tuple[str, int] = ("", 0)
    prev_scene: Optional[str] = None
    current_block_scene: Optional[str] = None
    current_block_len = 0

    for episode in _iter_episodes(path):
        total_episodes += 1
        scene_id = episode.get("scene_id", "")
        distances = _extract_agent_distances(episode)
        agent_count = len(episode.get("agents", []))
        agent_count_stats.update(float(agent_count))

        if len(distances) > 0:
            episode_mean_distance = sum(distances) / len(distances)
            scene_stats[scene_id].update(episode_mean_distance)
            scene_agent_counts[scene_id].update(float(agent_count))
            episode_mean_dist_stats.update(episode_mean_distance)

        if prev_scene is not None and scene_id != prev_scene:
            scene_transitions += 1

        if current_block_scene is None:
            current_block_scene = scene_id
            current_block_len = 1
        elif scene_id == current_block_scene:
            current_block_len += 1
        else:
            if len(first_blocks) < show_blocks:
                first_blocks.append((current_block_scene, current_block_len))
            if current_block_len > longest_block[1]:
                longest_block = (current_block_scene, current_block_len)
            current_block_scene = scene_id
            current_block_len = 1

        prev_scene = scene_id

    if current_block_scene is not None:
        if len(first_blocks) < show_blocks:
            first_blocks.append((current_block_scene, current_block_len))
        if current_block_len > longest_block[1]:
            longest_block = (current_block_scene, current_block_len)

    print(f"dataset: {path}")
    print(f"episodes: {total_episodes}")
    print(f"scenes: {len(scene_stats)}")
    print(
        "agents per episode: "
        f"mean={agent_count_stats.mean:.2f} "
        f"std={agent_count_stats.std:.2f} "
        f"min={agent_count_stats.min_value:.0f} "
        f"max={agent_count_stats.max_value:.0f}"
    )
    print(
        "episode mean geodesic distance: "
        f"mean={episode_mean_dist_stats.mean:.3f} "
        f"std={episode_mean_dist_stats.std:.3f} "
        f"min={episode_mean_dist_stats.min_value:.3f} "
        f"max={episode_mean_dist_stats.max_value:.3f}"
    )
    print(
        "scene order audit: "
        f"transitions={scene_transitions} "
        f"switch_rate={scene_transitions / max(total_episodes - 1, 1):.6f} "
        f"longest_block={_basename(longest_block[0])}:{longest_block[1]}"
    )
    if first_blocks:
        print("first scene blocks:")
        for scene_id, block_len in first_blocks:
            print(f"  {_basename(scene_id)} x {block_len}")

    per_scene_rows = []
    for scene_id, stats in scene_stats.items():
        per_scene_rows.append(
            (
                stats.mean,
                stats.std,
                stats.count,
                scene_agent_counts[scene_id].mean,
                scene_id,
            )
        )

    per_scene_rows.sort(key=lambda x: x[0])

    print(f"easiest {top_k} scenes by mean geodesic distance:")
    for mean_dist, std_dist, count, mean_agents, scene_id in per_scene_rows[:top_k]:
        print(
            f"  {_basename(scene_id)} "
            f"episodes={count} mean_dist={mean_dist:.3f} std={std_dist:.3f} "
            f"mean_agents={mean_agents:.2f}"
        )

    print(f"hardest {top_k} scenes by mean geodesic distance:")
    for mean_dist, std_dist, count, mean_agents, scene_id in per_scene_rows[-top_k:]:
        print(
            f"  {_basename(scene_id)} "
            f"episodes={count} mean_dist={mean_dist:.3f} std={std_dist:.3f} "
            f"mean_agents={mean_agents:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a multi-agent ImageNav dataset for scene ordering and difficulty."
    )
    parser.add_argument("dataset", help="Path to a .json.gz multi-agent dataset")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many easiest/hardest scenes to print",
    )
    parser.add_argument(
        "--show-blocks",
        type=int,
        default=10,
        help="How many scene blocks from file order to print",
    )
    args = parser.parse_args()
    analyze_dataset(args.dataset, top_k=args.top_k, show_blocks=args.show_blocks)


if __name__ == "__main__":
    main()
