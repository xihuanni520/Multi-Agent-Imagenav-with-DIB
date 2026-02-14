import habitat_sim
import numpy as np
import json
import gzip
import os
from tqdm import tqdm
import random

# --- 1. 定义官方标准划分列表 ---
TRAIN_SCENES = [
    "Adrian", "Albertville", "Anaheim", "Andover", "Angiola", "Annawan", "Applewold", 
    "Arkansaw", "Avonia", "Azusa", "Ballou", "Beach", "Bolton", "Bowlus", "Brevort", 
    "Capistrano", "Colebrook", "Convoy", "Cooperstown", "Crandon", "Delton", "Dryville", 
    "Dunmor", "Eagerville", "Goffs", "Hainesburg", "Hambleton", "Haxtun", "Hillsdale", 
    "Hometown", "Hominy", "Kerrtown", "Maryhill", "Mesic", "Micanopy", "Mifflintown", 
    "Mobridge", "Monson", "Mosinee", "Nemacolin", "Nicut", "Nimmons", "Nuevo", "Oyens", 
    "Parole", "Pettigrew", "Placida", "Pleasant", "Quantico", "Rancocas", "Reyno", 
    "Roane", "Roeville", "Rosser", "Roxboro", "Sanctuary", "Sasakwa", "Sawpit", "Seward", 
    "Shelbiana", "Silas", "Sodaville", "Soldier", "Spencerville", "Spotswood", "Springhill", 
    "Stanleyville", "Stilwell", "Stokes", "Sumas", "Superior"
]

VAL_SCENES = [
    "Cantwell", "Denmark", "Eastville", "Edgemere", "Elmira", "Eudora", "Greigsville", 
    "Mosquito", "Pablo", "Ribera", "Sands", "Scioto", "Sisters", "Swormville"
]

class MultiAgentDatasetGenerator:
    def __init__(self, scene_dir, save_dir):
        self.scene_dir = scene_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 配置参数 (你可以根据需要调整)
        self.min_geo_dist = 2.0    # 最小路径长度 (m)
        self.max_geo_dist = 20.0   # 最大路径长度 (m)
        self.agent_clearance = 1.5 # Agent之间出生点的最小间距 (m)
        self.max_retries = 2000    # 采样重试次数

    def _init_sim(self, scene_path):
        """初始化模拟器用于计算路径"""
        if not os.path.exists(scene_path):
            print(f"[Error] Scene file not found: {scene_path}")
            return None

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.load_semantic_mesh = False # 只需NavMesh，无需语义
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        try:
            sim = habitat_sim.Simulator(cfg)
            return sim
        except Exception as e:
            print(f"[Error] Failed to load {scene_path}: {e}")
            return None

    def _sample_agent_pose(self, sim, existing_starts=[]):
        """在NavMesh上采样合法的点，并避开已有的Agent"""
        pf = sim.pathfinder
        
        for _ in range(self.max_retries):
            pos = pf.get_random_navigable_point()
            
            # 检查碰撞 (Conflict Check)
            conflict = False
            for exist_p in existing_starts:
                if np.linalg.norm(np.array(pos) - np.array(exist_p)) < self.agent_clearance:
                    conflict = True
                    break
            
            if not conflict:
                # 随机生成 Y 轴旋转
                angle = np.random.uniform(0, 2 * np.pi)
                rotation = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)] 
                return pos, rotation
        return None, None

    def generate(self, split, num_agents, num_episodes_per_scene, mode='independent'):
        """
        生成数据集主函数
        :param split: 'train' 或 'val'
        :param num_agents: 智能体数量 N
        :param num_episodes_per_scene: 每个场景生成多少个 episode
        :param mode: 'independent' (各自不同目标) 或 'shared' (同一目标)
        """
        # 1. 根据 split 选择场景列表
        if split == 'train':
            target_scene_names = TRAIN_SCENES
        elif split == 'val':
            target_scene_names = VAL_SCENES
        else:
            raise ValueError("Split must be 'train' or 'val'")

        dataset = {"episodes": []}
        print(f"--- Generating {split.upper()} set: N={num_agents}, Mode={mode}, Scenes={len(target_scene_names)} ---")

        for scene_name in tqdm(target_scene_names):
            # 拼接完整路径
            scene_filename = f"{scene_name}.glb"
            scene_path = os.path.join(self.scene_dir, scene_filename)
            
            sim = self._init_sim(scene_path)
            if sim is None: continue
            
            pf = sim.pathfinder
            if not pf.is_loaded:
                print(f"[Warning] Pathfinder not loaded for {scene_name}")
                sim.close()
                continue

            # 开始为该场景生成 episodes
            episodes_generated = 0
            # 使用 while 循环确保生成足量的 valid episodes
            attempts = 0
            while episodes_generated < num_episodes_per_scene and attempts < num_episodes_per_scene * 5:
                attempts += 1
                
                episode_data = {
                    "scene_id": f"gibson/{scene_filename}", # Habitat 标准格式
                    "episode_id": f"{scene_name}_{episodes_generated}",
                    "agents": [],
                    "info": {"mode": mode, "split": split}
                }

                current_starts = []
                
                # Shared Goal 模式：先定一个共同目标
                shared_goal_pos = None
                if mode == 'shared':
                    shared_goal_pos = pf.get_random_navigable_point()

                agents_data = []
                success_all_agents = True

                for agent_id in range(num_agents):
                    # 1. 采样起点
                    start_pos, start_rot = self._sample_agent_pose(sim, current_starts)
                    if start_pos is None:
                        success_all_agents = False
                        break

                    # 2. 采样终点并验证路径
                    goal_pos = None
                    path_dist = 0.0
                    
                    # 尝试寻找合法目标 (距离阈值筛选)
                    retry_goal = 0
                    while retry_goal < 50:
                        if mode == 'shared':
                            temp_goal = shared_goal_pos
                        else:
                            temp_goal = pf.get_random_navigable_point()
                        
                        path = habitat_sim.ShortestPath()
                        path.requested_start = start_pos
                        path.requested_end = temp_goal
                        found_path = pf.find_path(path)
                        dist = path.geodesic_distance
                        
                        # 核心判定：是否可达且距离适中
                        if found_path and (self.min_geo_dist < dist < self.max_geo_dist):
                            goal_pos = temp_goal
                            path_dist = dist
                            break
                        
                        if mode == 'shared': 
                            # 共享目标如果不通，则整个episode作废
                            break 
                        retry_goal += 1

                    if goal_pos is None:
                        success_all_agents = False
                        break

                    # 记录 Agent 数据
                    current_starts.append(start_pos)
                    agents_data.append({
                        "agent_id": agent_id,
                        "start_position": [float(x) for x in start_pos],
                        "start_rotation": [float(x) for x in start_rot],
                        "goals": [{
                            "position": [float(x) for x in goal_pos],
                            "radius": None
                        }],
                        "info": {
                            "geodesic_distance": float(path_dist)
                        }
                    })

                if success_all_agents:
                    episode_data["agents"] = agents_data
                    dataset["episodes"].append(episode_data)
                    episodes_generated += 1
            
            sim.close()

        # 保存结果
        filename = f"{self.save_dir}/{split}_{num_agents}_agents_{mode}.json.gz"
        print(f"Saving {len(dataset['episodes'])} episodes to {filename} ...")
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(dataset, f)
        print("Done!")
        return filename

# --- 使用说明 ---
if __name__ == "__main__":
    # 1. 配置路径
    # 请确保你的 .glb 文件就在这个目录下
    SCENE_DIR = "../data/scene_datasets/gibson" 
    
    # 输出目录
    SAVE_DIR_TRAIN = "../data/datasets/multinav/gibson/v1/train/content"
    SAVE_DIR_VAL = "../data/datasets/multinav/gibson/v1/val/content"

    # 2. 实例化生成器
    # 注意：这里我们传入 SAVE_DIR_TRAIN 只是默认，实际保存路径在 generate 里可以灵活改，
    # 但为了简单，我们创建两个实例或者在外部控制
    
    # 生成 TRAIN 集 (72个场景)
    gen_train = MultiAgentDatasetGenerator(SCENE_DIR, SAVE_DIR_TRAIN)
    # 例如：生成 3 个 Agent，每个场景 100 个 Episode = 总共 7200 个 Episode
    gen_train.generate(split='train', num_agents=4, num_episodes_per_scene=9000, mode='independent')

    # 生成 VAL 集 (14个场景)
    gen_val = MultiAgentDatasetGenerator(SCENE_DIR, SAVE_DIR_VAL)
    # 例如：生成 3 个 Agent，每个场景 20 个 Episode = 总共 280 个 Episode
    gen_val.generate(split='val', num_agents=4, num_episodes_per_scene=300, mode='independent')