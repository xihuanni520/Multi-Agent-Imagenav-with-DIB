import os
import json
import gzip
import glob

def get_scene_names(split_path):
    """
    从数据集中提取唯一的场景 ID 并转换为场景名
    """
    scenes = set()
    if not os.path.exists(split_path):
        return scenes
        
    # 读取所有 json.gz 文件
    files = glob.glob(os.path.join(split_path, "*.json.gz"))
    
    for file_path in files:
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                # 遍历 episode 获取 scene_id
                for ep in data['episodes']:
                    # scene_id 示例: "data/scene_datasets/gibson/Allensville.glb"
                    full_path = ep['scene_id']
                    # 提取文件名 "Allensville"
                    scene_name = os.path.basename(full_path).replace('.glb', '')
                    scenes.add(scene_name)
                    
                    # 某些拆分文件(content)只包含一个场景，提取到一个就可以跳出当前文件
                    if 'content' in split_path:
                        break 
        except Exception as e:
            print(f"读取错误 {file_path}: {e}")
            
    return sorted(list(scenes))

if __name__ == "__main__":
    # 定义路径 [Source 18]
    train_path = "../data/datasets/imagenav/gibson/v1/train/content"
    val_path = "../data/datasets/multinav/gibson/v1/val"

    print("正在读取训练集场景...")
    train_scenes = get_scene_names(train_path)
    
    print("正在读取验证集场景...")
    val_scenes = get_scene_names(val_path)

    print(f"\n{'='*40}")
    print(f"训练集场景总数: {len(train_scenes)} (预期: 72)")
    print(f"验证集场景总数: {len(val_scenes)} (预期: 14)")
    print(f"{'='*40}\n")

    print(f"【验证集 (14个)】:")
    print(", ".join(val_scenes))
    
    print(f"\n【训练集 (前10个示例)】:")
    print(", ".join(train_scenes[:-1]) + " ...")