import os
import gzip
import json
import numpy as np

def analyze_dataset_difficulty(content_dir):
    """
    统计 Habitat 数据集中每个场景的平均测地距离（任务难度）。
    """
    if not os.path.exists(content_dir):
        print(f"[Error] Directory not found: {content_dir}")
        return

    # 获取所有场景的 json.gz 文件
    # 过滤掉可能的非场景文件，比如 manifest 文件
    scene_files = [f for f in os.listdir(content_dir) if f.endswith('.json.gz') and f != 'train.json.gz' and f != 'val.json.gz']
    
    print(f"Found {len(scene_files)} scene files in {content_dir}. Starting analysis...\n")
    
    results = []

    for filename in scene_files:
        filepath = os.path.join(content_dir, filename)
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'episodes' not in data:
                continue
                
            distances = []
            for ep in data['episodes']:
                # Habitat 单智能体标准数据集的测地距离通常存在 info 里
                if 'info' in ep and 'geodesic_distance' in ep['info']:
                    distances.append(ep['info']['geodesic_distance'])
            
            if len(distances) > 0:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                # 将 .json.gz 替换为 .glb 匹配你的输出格式要求
                scene_name = filename.replace('.json.gz', '.glb') 
                
                results.append({
                    'scene': scene_name,
                    'episodes': len(distances),
                    'mean': mean_dist,
                    'std': std_dist
                })
                
        except Exception as e:
            print(f"[Warning] Failed to process {filename}: {e}")

    # 按照 mean_dist 从小到大排序（从易到难）
    results.sort(key=lambda x: x['mean'])

    # 打印结果
    print("-" * 60)
    print(f"{'Scene Name':<20} | {'Episodes':<10} | {'Mean Dist':<10} | {'Std Dev':<10}")
    print("-" * 60)
    
    for r in results:
        # 格式化输出，保留三位小数
        print(f"{r['scene']:<20} episodes={r['episodes']:<8} mean_dist={r['mean']:.3f}  std={r['std']:.3f}")

if __name__ == "__main__":
    # 请修改为你的官方 Gibson ImageNav 数据集 content 目录的实际路径
    # 通常的路径结构是: data/datasets/imagenav/gibson/v1/train/content
    DATASET_CONTENT_DIR = "../data/datasets/imagenav/gibson/v1/train/content"
    
    analyze_dataset_difficulty(DATASET_CONTENT_DIR)