import habitat_sim
import numpy as np
import json
import gzip
import os
from tqdm import tqdm
import random

class MultiAgentDatasetGenerator:
    def __init__(self, scene_dir, save_dir):
        self.scene_dir = scene_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 配置参数
        self.min_geo_dist = 2.0   # 最小路径长度（米）
        self.max_geo_dist = 20.0  # 最大路径长度
        self.agent_clearance = 1.5 # Agent之间出生点的最小间距（避免重叠）
        self.max_retries = 1000   # 采样失败重试次数

    def _init_sim(self, scene_path):
        """初始化模拟器用于计算路径"""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        # 不渲染图像，只加载NavMesh，速度极快
        sim_cfg.load_semantic_mesh = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        try:
            sim = habitat_sim.Simulator(cfg)
            return sim
        except Exception as e:
            print(f"Failed to load {scene_path}: {e}")
            return None

    def _sample_agent_pose(self, sim, existing_starts=[]):
        """在NavMesh上采样合法的点"""
        pf = sim.pathfinder
        
        for _ in range(self.max_retries):
            # 随机采样一个点
            pos = pf.get_random_navigable_point()
            
            # 检查与已有的Agent出生点是否冲突
            conflict = False
            for exist_p in existing_starts:
                # 欧氏距离判断重叠
                if np.linalg.norm(np.array(pos) - np.array(exist_p)) < self.agent_clearance:
                    conflict = True
                    break
            if not conflict:
                # 随机生成朝向 (quaternion)
                angle = np.random.uniform(0, 2 * np.pi)
                # Habitat中使用 [0, sin(a/2), 0, cos(a/2)] 代表绕Y轴旋转
                rotation = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)] 
                return pos, rotation
        return None, None

    def generate(self, num_agents, num_episodes_per_scene, mode='independent', scenes=None):
        """
        核心生成逻辑
        :param num_agents: Agent数量 N
        :param mode: 'independent' (各自不同目标) 或 'shared' (同一目标)
        """
        dataset = {"episodes": []}
        
        if scenes is None:
            # 默认读取Gibson文件夹下的glb
            scenes = [f for f in os.listdir(self.scene_dir) if f.endswith('.glb')]
            scenes.sort()

        print(f"Start generating: N={num_agents}, Mode={mode}, Scenes={len(scenes)}")

        for scene_file in tqdm(scenes):
            scene_path = os.path.join(self.scene_dir, scene_file)
            sim = self._init_sim(scene_path)
            if sim is None: continue
            
            pf = sim.pathfinder
            if not pf.is_loaded:
                print(f"Pathfinder not loaded for {scene_file}")
                sim.close()
                continue

            for ep_idx in range(num_episodes_per_scene):
                episode_data = {
                    "scene_id": f"gibson/{scene_file}", # 对应Habitat读取路径
                    "episode_id": f"{scene_file}_{ep_idx}",
                    "agents": [],
                    "info": {"mode": mode}
                }

                current_starts = []
                
                # --- Shared Goal 模式预先采样目标 ---
                shared_goal_pos = None
                if mode == 'shared':
                    shared_goal_pos = pf.get_random_navigable_point()

                # --- 为 N 个 Agent 循环采样 ---
                agents_data = []
                success_all_agents = True

                for agent_id in range(num_agents):
                    # 1. 采样起点 (保证不重叠)
                    start_pos, start_rot = self._sample_agent_pose(sim, current_starts)
                    if start_pos is None:
                        success_all_agents = False
                        break

                    # 2. 确定终点
                    goal_pos = None
                    path_dist = 0.0

                    retry_goal = 0
                    while retry_goal < 100:
                        if mode == 'shared':
                            temp_goal = shared_goal_pos
                        else: # independent
                            temp_goal = pf.get_random_navigable_point()
                        
                        # 3. 验证连通性和距离
                        path = habitat_sim.ShortestPath()
                        path.requested_start = start_pos
                        path.requested_end = temp_goal
                        found_path = pf.find_path(path)
                        
                        dist = path.geodesic_distance
                        
                        if found_path and (self.min_geo_dist < dist < self.max_geo_dist):
                            goal_pos = temp_goal
                            path_dist = dist
                            break
                        
                        if mode == 'shared': 
                            # 如果是共享目标但这个起点走不到目标，则需要重新采样起点
                            # 简单起见，这里如果shared模式失败，直接break外层重来
                            break 
                        retry_goal += 1

                    if goal_pos is None:
                        success_all_agents = False
                        break

                    # 记录这个Agent的数据
                    current_starts.append(start_pos)
                    agents_data.append({
                        "agent_id": agent_id,
                        "start_position": [float(x) for x in start_pos],
                        "start_rotation": [float(x) for x in start_rot],
                        "goals": [{
                            "position": [float(x) for x in goal_pos],
                            "radius": None # ImageNav通常不需要半径
                        }],
                        "info": {
                            "geodesic_distance": float(path_dist)
                        }
                    })

                if success_all_agents:
                    episode_data["agents"] = agents_data
                    dataset["episodes"].append(episode_data)
            
            sim.close()

        # 保存
        filename = f"{self.save_dir}/train_{num_agents}_agents_{mode}.json.gz"
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(dataset, f)
        print(f"Saved {len(dataset['episodes'])} episodes to {filename}")
        return dataset

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置你的路径
    SCENE_DIR = "../data/scene_datasets/gibson"  # 你的.glb文件所在位置
    # SAVE_DIR = "../data/datasets/multinav/gibson/v1/train/content/test_generate"
    SAVE_DIR = "../data/datasets/multinav/gibson/v1/train/content"
    
    gen = MultiAgentDatasetGenerator(SCENE_DIR, SAVE_DIR)
    
    # 生成 N=3 的独立目标数据集
    # data_3_indep = gen.generate(num_agents=3, num_episodes_per_scene=1, mode='independent')
    data_3_indep = gen.generate(num_agents=4, num_episodes_per_scene=50, mode='independent')
    
    # 如果你想生成 N=4 的共享目标数据集
    # gen.generate(num_agents=4, num_episodes_per_scene=50, mode='shared')