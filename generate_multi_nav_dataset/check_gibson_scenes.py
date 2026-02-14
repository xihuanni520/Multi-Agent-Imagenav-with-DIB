import os
import json
import gzip
import glob

def count_train_scenes(train_path):
    """
    统计训练集 content 目录下的文件数量（通常 1 文件 = 1 场景）
    """
    if not os.path.exists(train_path):
        print(f"❌ 路径不存在: {train_path}")
        return

    # 获取所有 .json.gz 文件
    files = glob.glob(os.path.join(train_path, "*.json.gz"))
    file_count = len(files)
    
    print(f"-" * 30)
    print(f"📊 训练集统计 (Train)")
    print(f"   路径: {train_path}")
    print(f"   检测到的场景文件数: {file_count}")
    # 打印前3个文件名作为示例
    if file_count > 0:
        print(f"   示例文件: {[os.path.basename(f) for f in files[:3]]} ...")

def count_val_unique_scenes(val_path):
    """
    解析验证集 JSON 文件，统计内部实际利用的 Unique Scenes
    """
    if not os.path.exists(val_path):
        print(f"❌ 路径不存在: {val_path}")
        return

    files = glob.glob(os.path.join(val_path, "*.json.gz"))
    unique_scenes = set()
    total_episodes = 0

    print(f"-" * 30)
    print(f"📊 验证集统计 (Val)")
    print(f"   路径: {val_path}")
    
    for file_path in files:
        print(f"   正在读取: {os.path.basename(file_path)} ...")
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                episodes = data.get('episodes', [])
                total_episodes += len(episodes)
                
                # 遍历所有 episode 提取 scene_id
                for ep in episodes:
                    # scene_id 通常格式为 "data/scene_datasets/gibson/Allensville.glb"
                    full_scene_path = ep.get('scene_id', '')
                    # 我们只提取场景名，比如 "Allensville"
                    scene_name = os.path.basename(full_scene_path).replace('.glb', '')
                    unique_scenes.add(scene_name)
        except Exception as e:
            print(f"   ⚠️ 读取文件出错: {e}")

    print(f"   ✅ 验证集包含的总 Episode 数: {total_episodes}")
    print(f"   ✅ 验证集覆盖的唯一场景数 (Unique Scenes): {len(unique_scenes)}")
    print(f"   场景列表示例: {list(unique_scenes)[:-1]} ...")

if __name__ == "__main__":
    # 定义 FGPrompt 的数据路径 [Source 18]
    train_dir = "../data/datasets/imagenav/gibson/v1/train/content"
    val_dir = "../data/datasets/multinav/gibson/v1/val/content"

    count_train_scenes(train_dir)
    count_val_unique_scenes(val_dir)