import gzip
import json
import os

# 假设你的数据路径如下（根据你的实际解压位置修改）
# 注意：Gibson的train通常被拆分在 content 文件夹里的多个小文件中
# dataset_path = "../data/datasets/imagenav/gibson/v1/train/content"
dataset_path = "../data/datasets/imagenav/gibson/v1/val"

# 遍历该目录下的所有文件
if os.path.exists(dataset_path):
    print(f"正在检查目录: {dataset_path}")
    files = [f for f in os.listdir(dataset_path) if f.endswith('.json.gz')]
    
    total_episodes = 0
    for file_name in files:
        file_path = os.path.join(dataset_path, file_name)
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            # data['episodes'] 是一个列表，长度就是该场景的 episode 数量
            count = len(data['episodes'])
            print(f"场景文件: {file_name} -> 包含 {count} 个 episodes")
            total_episodes += count
            
            
    print(f"总计训练 Episodes: {total_episodes}")
else:
    print("找不到数据目录，请确认 data/datasets/imagenav/gibson/v1/train/content 路径是否正确")