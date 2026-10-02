#!/usr/bin/env bash
# 5个已通过本轮teacher入库筛选的独立GUI回放命令。
# 先激活训练teacher所用的Python环境，在whole_body_tracking仓库根目录执行。
# 所有执行命令保持注释：复制一条命令并去掉开头的“# ”，关闭窗口后再运行下一条。
# 每条只显示1个G1，使用该进程的cuda:0；如需选物理显卡，在命令前添加CUDA_VISIBLE_DEVICES=卡号。
# 沿用仓库默认seed；不设置CUDA版本，不带--headless，不自动启动回放。
# 默认play保留原task的观测噪声、物理随机化、随机起始phase和10秒episode重置。
# 这是可视化检查，不等同于phase0到motion末尾的G1完整回放验收。
# 该入口会先导出ONNX到对应训练目录的exported/，然后持续回放到关闭窗口。
# teacher数量够不够取决于技能覆盖：这5个可先做stage2多动作基线，后续需补齐跑步/冲刺和倒地恢复。

# 1. walk1_subject1：完整clean 20/20；随机phase扰动10秒 20/20。
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --experiment_name g1_flat --load_run 2026-10-01_15-41-59_walk1_subject1_default --checkpoint model_29999.pt --motion_file data/reproduction/motions_25hz/walk1_subject1.npz --num_envs 1 --device cuda:0

# 2. walk2_subject1：完整clean 20/20；随机phase扰动10秒 20/20。
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --experiment_name g1_flat --load_run 2026-10-01_15-42-05_walk2_subject1_default --checkpoint model_29999.pt --motion_file data/reproduction/motions_25hz/walk2_subject1.npz --num_envs 1 --device cuda:0

# 3. dance1_subject1：完整clean 20/20；随机phase扰动10秒 19/20。
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --experiment_name g1_flat --load_run 2026-10-01_15-44-46_dance1_subject1_default --checkpoint model_29999.pt --motion_file data/reproduction/motions_25hz/dance1_subject1.npz --num_envs 1 --device cuda:0

# 4. jumps1_subject1：完整clean 20/20；随机phase扰动10秒 20/20。
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --experiment_name g1_flat --load_run 2026-10-01_16-15-06_jumps1_subject1_default --checkpoint model_29999.pt --motion_file data/reproduction/motions_25hz/jumps1_subject1.npz --num_envs 1 --device cuda:0

# 5. fight1_subject2：完整clean 20/20；随机phase扰动10秒 19/20。
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --experiment_name g1_flat --load_run 2026-10-01_16-15-39_fight1_subject2_default --checkpoint model_29999.pt --motion_file data/reproduction/motions_25hz/fight1_subject2.npz --num_envs 1 --device cuda:0
