import matplotlib.pyplot as plt
import numpy as np
import gzip
import json

def plot_evaluation(dataset_paths_dict):
    """
    dataset_paths_dict: {N: path_to_json_gz}
    例如: {2: 'path/to/2_agents.json.gz', 3: 'path/to/3_agents.json.gz'}
    """
    
    # 存储统计数据
    stats = {} # Key: N, Value: list of all distances
    
    for n_agents, path in dataset_paths_dict.items():
        print(f"Loading N={n_agents} from {path}...")
        distances = []
        try:
            with gzip.open(path, 'rt') as f:
                data = json.load(f)
                
            for ep in data['episodes']:
                # 收集该episode中所有agent的平均距离，或者所有agent的距离
                # 题目要求：每回合平均轨迹长度（每个agent的平均轨迹长度）
                agent_dists = [a['info']['geodesic_distance'] for a in ep['agents']]
                avg_ep_dist = np.mean(agent_dists)
                distances.append(avg_ep_dist)
                
            stats[n_agents] = distances
        except Exception as e:
            print(f"Error loading {path}: {e}")

    # --- 绘图 1: 轨迹长度分布图 (以 N=3 为例，或者画所有) ---
    plt.figure(figsize=(10, 6))
    
    # 这里我们画出所有N的分布，或者只画你指定的某一个主要N
    target_n = list(stats.keys())[0] # 取第一个N来画直方图
    data_to_hist = stats[target_n]
    
    plt.hist(data_to_hist, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Average Episode Trajectory Length (N={target_n})")
    plt.xlabel("Geodesic Distance (m)")
    plt.ylabel("Episode Count")
    plt.grid(axis='y', alpha=0.5)
    
    # 插入图片占位符 (如果需要动态生成图片，此处为逻辑)
    # 
    plt.savefig("plot1_dist_distribution_new_3.png")
    plt.show()

    # --- 绘图 2: N vs 平均最短路径长度 ---
    plt.figure(figsize=(8, 5))
    
    x_agents = sorted(stats.keys())
    y_avg_dists = [np.mean(stats[n]) for n in x_agents]
    y_std_dists = [np.std(stats[n]) for n in x_agents] # 也可以画标准差
    
    plt.plot(x_agents, y_avg_dists, marker='o', linestyle='-', linewidth=2, color='orange')
    plt.errorbar(x_agents, y_avg_dists, yerr=y_std_dists, fmt='o', color='orange', capsize=5, label='Std Dev')
    
    plt.title("Average Path Length vs. Number of Agents")
    plt.xlabel("Number of Agents (N)")
    plt.ylabel("Avg Geodesic Distance (m)")
    plt.xticks(x_agents)
    plt.legend()
    plt.grid(True)
    
    # 
    plt.savefig("plot2_n_vs_dist.png")
    plt.show()

# --- 模拟运行绘图逻辑 ---
# 假设你已经用上面的生成器生成了 N=2,3,4 的数据集
# 你可以这样调用：
paths = {
#    2: "data/datasets/multinav/gibson/v1/train/content/train_2_agents_independent.json.gz",
   3: "../data/datasets/multinav/gibson/v1/train/content/train_3_agents_independent.json.gz",
   4: "../data/datasets/multinav/gibson/v1/train/content/train_4_agents_independent.json.gz"
}
plot_evaluation(paths)