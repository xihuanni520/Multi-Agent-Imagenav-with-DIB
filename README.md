# Multi-Agent ImageNav with DIB

This repository extends **FGPrompt** from single-agent ImageNav to **multi-agent ImageNav** with communication-ready training logic.

## Overview

- Base backbone: FGPrompt early-fusion idea (observation RGB + goal RGB channel-wise fusion).
- Task: in each episode, `N` agents navigate to their own image goals in the same scene.
- Multi-agent synchronization: if one agent succeeds early, it stays in place until all agents in the same group finish or hit max steps.
- Communication design (DIB-style): each sender compresses local observation features and shares them with teammates; receivers fuse incoming messages with their own early-fused features before policy inference.

## What Is Different from Original FGPrompt

- Added multi-agent dataset support (`MultiImageNav-v1`).
- Added multi-agent environment execution and group-level synchronization.
- Added multi-agent metrics:
  - per-agent success/SPL/distance
  - `success_mean`
  - `success_all`
- Kept FGPrompt RL pipeline and policy family as the core baseline.

## Installation

Follow the same dependency stack as FGPrompt (Habitat-Lab + Habitat-Baselines + PyTorch 1.11 + Habitat-Sim 0.2.2).

```bash
git clone https://github.com/XinyuSun/FGPrompt.git
cd FGPrompt

git submodule init
git submodule update

conda create -n fgprompt python=3.8
conda activate fgprompt

conda install habitat-sim=0.2.2 withbullet headless -c conda-forge -c aihabitat
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

cd habitat-lab
git checkout 1f7cfbdd3debc825f1f2fd4b9e1a8d6d4bc9bfc7
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..

pip install -r requirements.txt
```

## Multi-Agent Dataset Format

Expected episode format:

```json
{
  "episodes": [
    {
      "scene_id": "gibson/xxx.glb",
      "episode_id": "xxx_0",
      "agents": [
        {
          "agent_id": 0,
          "start_position": [x, y, z],
          "start_rotation": [x, y, z, w],
          "goals": [{"position": [x, y, z], "radius": null}],
          "info": {"geodesic_distance": d}
        }
      ],
      "info": {"mode": "independent"}
    }
  ]
}
```

Current config expects:

```text
/home/cyf/FGPrompt/data/datasets/multinav/gibson/v1/train/content/train_{num_agents}_agents_independent.json.gz
```

Scenes should be available under:

```text
/home/cyf/FGPrompt/data/scene_datasets/gibson
```

## Train

Example (2 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 29501 --nproc_per_node=2 \
run.py \
  --exp-config exp_config/ddppo_imagenav_gibson_multi.yaml,policy,reward_multi,early-fusion \
  --run-type train
```

## Eval

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29502 --nproc_per_node=1 \
run.py \
  --exp-config exp_config/ddppo_imagenav_gibson_multi.yaml,policy,reward_multi,early-fusion,eval \
  --run-type eval
```

## Notes

- This codebase is under active development for multi-agent communication and efficiency.
- For reproducibility, keep config, dataset split, and scene assets fixed across runs.

## Acknowledgement

This project is built on top of:

- **FGPrompt**: Fine-grained Goal Prompting for Image-goal Navigation
- **Habitat-Lab / Habitat-Baselines**

We sincerely thank the FGPrompt authors for open-sourcing their codebase and providing a strong foundation for this multi-agent extension.

## Citation

If you use this repository, please cite **FGPrompt** first:

```bibtex
@inproceedings{fgprompt2023,
  author = {Xinyu, Sun and Peihao, Chen and Jugang, Fan and Thomas, H. Li and Jian, Chen and Mingkui, Tan},
  title = {FGPrompt: Fine-grained Goal Prompting for Image-goal Navigation},
  booktitle = {37th Conference on Neural Information Processing Systems (NeurIPS 2023)},
  year = {2023}
}
```
