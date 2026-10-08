# 从 LingBot 基础权重训练到 5000 步

本目录提供 clean-only 训练和 clean 评测代码。训练从原始 LingBot-VLA 预训练权重开始，默认跑 5000 个 optimizer step，每 1000 步保存一次。数据集、模型权重和仿真资源不包含在内。

## 已测结果（仅 clean）

两次评测都使用 Aloha-AgileX、unseen 指令、fp32、动作块 50、执行长度 50、OptiX 渲染。回合数不同，分数不能直接相减。

| 模型 | 协议 | 成功 |
|---|---|---:|
| LingBot 发布后训练权重 [model_link](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin) | 50 任务 × 100 回合 | 3813/5000（76.26%） |
| 原始预训练权重继续训练到 5000 步 | 50 任务 × 10 回合 | 269/500（53.80%） |

※ 与官方结果存在差别的原因：该代码训练使用的是clean数据计算的边界和lingbot-vla-v2-6b-robotwin这个后训练边界权重不一致，测评使用的是clean边界数据，映射存在误差。

基础 LingBot 指已完整评测的发布权重。训练起点是原始预训练权重，不是这份发布权重。原始预训练权重的全量评测没有跑完，这里不记它的分数。5000 步结果来自 clean 示范、帧均匀采样、Muon、学习率 1e-4、seed 42、3 卡、micro-batch 8、global batch 120。

| 任务 | 基础 LingBot | 比例 | 5000 步 | 比例 |
|---|---:|---:|---:|---:|
| adjust_bottle | 100/100 | 100% | 10/10 | 100% |
| beat_block_hammer | 70/100 | 70% | 7/10 | 70% |
| blocks_ranking_rgb | 81/100 | 81% | 7/10 | 70% |
| blocks_ranking_size | 77/100 | 77% | 0/10 | 0% |
| click_alarmclock | 17/100 | 17% | 2/10 | 20% |
| click_bell | 6/100 | 6% | 10/10 | 100% |
| dump_bin_bigbin | 89/100 | 89% | 9/10 | 90% |
| grab_roller | 100/100 | 100% | 10/10 | 100% |
| handover_block | 54/100 | 54% | 1/10 | 10% |
| handover_mic | 100/100 | 100% | 4/10 | 40% |
| hanging_mug | 21/100 | 21% | 1/10 | 10% |
| lift_pot | 87/100 | 87% | 2/10 | 20% |
| move_can_pot | 81/100 | 81% | 5/10 | 50% |
| move_pillbottle_pad | 85/100 | 85% | 4/10 | 40% |
| move_playingcard_away | 100/100 | 100% | 9/10 | 90% |
| move_stapler_pad | 27/100 | 27% | 2/10 | 20% |
| open_laptop | 98/100 | 98% | 7/10 | 70% |
| open_microwave | 38/100 | 38% | 2/10 | 20% |
| pick_diverse_bottles | 43/100 | 43% | 3/10 | 30% |
| pick_dual_bottles | 58/100 | 58% | 4/10 | 40% |
| place_a2b_left | 90/100 | 90% | 5/10 | 50% |
| place_a2b_right | 86/100 | 86% | 6/10 | 60% |
| place_bread_basket | 87/100 | 87% | 6/10 | 60% |
| place_bread_skillet | 73/100 | 73% | 8/10 | 80% |
| place_burger_fries | 78/100 | 78% | 8/10 | 80% |
| place_can_basket | 81/100 | 81% | 6/10 | 60% |
| place_cans_plasticbox | 99/100 | 99% | 1/10 | 10% |
| place_container_plate | 91/100 | 91% | 9/10 | 90% |
| place_dual_shoes | 71/100 | 71% | 3/10 | 30% |
| place_empty_cup | 93/100 | 93% | 8/10 | 80% |
| place_fan | 53/100 | 53% | 1/10 | 10% |
| place_mouse_pad | 76/100 | 76% | 1/10 | 10% |
| place_object_basket | 72/100 | 72% | 7/10 | 70% |
| place_object_scale | 97/100 | 97% | 5/10 | 50% |
| place_object_stand | 95/100 | 95% | 9/10 | 90% |
| place_phone_stand | 92/100 | 92% | 7/10 | 70% |
| place_shoe | 92/100 | 92% | 7/10 | 70% |
| press_stapler | 93/100 | 93% | 10/10 | 100% |
| put_bottles_dustbin | 70/100 | 70% | 0/10 | 0% |
| put_object_cabinet | 64/100 | 64% | 4/10 | 40% |
| rotate_qrcode | 81/100 | 81% | 6/10 | 60% |
| scan_object | 26/100 | 26% | 4/10 | 40% |
| shake_bottle | 99/100 | 99% | 10/10 | 100% |
| shake_bottle_horizontally | 100/100 | 100% | 10/10 | 100% |
| stack_blocks_three | 97/100 | 97% | 1/10 | 10% |
| stack_blocks_two | 99/100 | 99% | 7/10 | 70% |
| stack_bowls_three | 83/100 | 83% | 3/10 | 30% |
| stack_bowls_two | 96/100 | 96% | 6/10 | 60% |
| stamp_seal | 89/100 | 89% | 3/10 | 30% |
| turn_switch | 58/100 | 58% | 9/10 | 90% |

## 目录

```
scripts/setup_env.sh      写入本机路径并检查环境
scripts/train.sh          一键训练到 5000 步
scripts/eval.sh           一键 clean 评测
scripts/retarget_index.py 把数据索引改到当前数据目录
train/                    训练配置与训练器
eval/                     评测调度、推理服务和仿真客户端
.cursor/skills/setup-train-env/SKILL.md
```

## 需要自行放置的内容

不要把这些内容打进代码包。默认位置都相对本目录：

| 用途 | 路径 |
|---|---|
| 训练集 | `data/robotwin2_aloha_clean/` |
| 训练起点 | `models/lingbot-vla-v2-6b/` |
| 文本与视觉 tokenizer | `models/Qwen3-VL-4B-Instruct/` |
| 深度教师 | `models/moge-2-vitb-normal/model.pt` 和 `models/lingbot-vla-v2-6b/depth/model.pt` |
| 视频教师 | `models/lingbot-vla-v2-6b/dino_video/` |
| 模型源码 | `third_party/lingbot-vla-v2/` |
| 仿真源码 | `third_party/robotwin/` |

训练集是 RoboTwin 2.0 的 Aloha-AgileX clean，50 个任务、每任务 50 条。每个任务目录需要 `episode0` 到 `episode49` 的 `.hdf5`、`.rgb`、`.npz`，以及一份 `data_index.json`。索引只使用 seen 指令。若索引里的路径来自别的机器，先执行：

```bash
source scripts/env.sh
"$VLA_TRAIN_PYTHON" scripts/retarget_index.py
```

## 一键配置和运行

环境配置说明在 `.cursor/skills/setup-train-env/SKILL.md`。

```bash
bash scripts/setup_env.sh --train-python /path/to/train-python --sim-python /path/to/sim-python
# 编辑 env.local.sh，设置空闲的 VLA_GPU_IDS
bash scripts/setup_env.sh --check
bash scripts/train.sh
bash scripts/eval.sh --checkpoint results/<run>/checkpoints/global_step_5000/hf_ckpt
```

`train.sh` 在所选 GPU 已有计算进程时会拒绝启动。评测默认 50 个任务、每任务 10 条 clean、执行长度 50，最多使用 5 张卡。OptiX 去噪需要系统里存在 `/usr/share/nvidia/nvoptix.bin`，以及 NVIDIA 的 GLX、EGL 和 `libnvoptix`。

训练输出写到 `results/baseline_<时间>/`。评测输出写到 `results/<时间>_execute50/`，汇总见该目录的 `metrics.md`。
