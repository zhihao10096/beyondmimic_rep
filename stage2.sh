#!/usr/bin/env bash
# 在仓库根目录、原teacher训练环境中执行；命令全部保留注释，逐条复制执行。
# 显卡由你分配：可在命令前加CUDA_VISIBLE_DEVICES=卡号；不指定CUDA版本。
# workflow.py自动读取teacher_registry_current.json中的权重、motion、YAML和验收报告。
# 每个采集命令依次完成train/val独立采集；输出已存在时拒绝覆盖。
# 分服务器采集时，训练前把全部入库motion的数据汇总到训练机data/stage2，并复制当前登记表。

# ======================================================================
# 第一类：使用现有5个合格teacher，开始第二阶段训练
# ======================================================================

# 1. 采集D0：teacher执行。先跑walk1检查输出，再启动其他4项；可分配到不同卡。
# 每项train=32环境×2500步，val=32环境×1000步；独立采集seed100/200。
# 输出：data/stage2/动作名/d0_train_s100和d0_val_s200。
# python scripts/stage2/workflow.py collect-d0 --motion walk1_subject1
# python scripts/stage2/workflow.py collect-d0 --motion walk2_subject1
# python scripts/stage2/workflow.py collect-d0 --motion dance1_subject1
# python scripts/stage2/workflow.py collect-d0 --motion jumps1_subject1
# python scripts/stage2/workflow.py collect-d0 --motion fight1_subject2

# 2. 上面5项train/val全部完成后，共同训练一个CVAE。
# 默认50 epochs、每epoch采样100000行、batch512，这是起始预算，以闭环结果判断收敛。
# 产物：logs/stage2/cvae_d0/best.pt和last.pt。
# python scripts/stage2/workflow.py train-d0

# 3. 完整回放全部入库motion，查看打印的summary路径及G2结果。
# 即使离线MSE很低，也应做下一步DAgger，通过实际student状态补充数据。
# python scripts/stage2/workflow.py evaluate --student logs/stage2/cvae_d0/best.pt

# 4. 第一轮DAgger：student执行，teacher在student实际状态上标注；每项自动采集train/val。
# 输出：data/stage2/动作名/d1_train_s101和d1_val_s201。
# python scripts/stage2/workflow.py collect-dagger --motion walk1_subject1 --student logs/stage2/cvae_d0/best.pt --round 1
# python scripts/stage2/workflow.py collect-dagger --motion walk2_subject1 --student logs/stage2/cvae_d0/best.pt --round 1
# python scripts/stage2/workflow.py collect-dagger --motion dance1_subject1 --student logs/stage2/cvae_d0/best.pt --round 1
# python scripts/stage2/workflow.py collect-dagger --motion jumps1_subject1 --student logs/stage2/cvae_d0/best.pt --round 1
# python scripts/stage2/workflow.py collect-dagger --motion fight1_subject2 --student logs/stage2/cvae_d0/best.pt --round 1

# 5. 聚合D0+D1继续训练；产物：logs/stage2/cvae_d1/best.pt。
# python scripts/stage2/workflow.py train-dagger --student logs/stage2/cvae_d0/best.pt --round 1
# python scripts/stage2/workflow.py evaluate --student logs/stage2/cvae_d1/best.pt
# 若未达到G2，重复第4—5步：student换为上一轮best.pt，round改为2、3……；旧轮次数据自动保留。
# 也可使用自动脚本，从现有D1开始继续D2、D3……，直到全部motion通过G2 clean：
# bash dagger.sh
# 说明见docs/beyondmimic_reproduction/14_自动DAgger.md。

# 6. G2 clean合格后，补做完整动作扰动回放；独立报告。
# python scripts/stage2/workflow.py evaluate --student logs/stage2/cvae_d1/best.pt --perturbed
# 确定最终CVAE后导出decoder，供第三阶段使用。
# python scripts/stage2/export.py --checkpoint logs/stage2/cvae_d1/best.pt --output artifacts/stage2/decoder.pt

# ======================================================================
# 第二类：扩充teacher时，需要做哪些工作
# ======================================================================

# A. 已有但未入库的teacher：复制更晚checkpoint及原params到对应训练目录，再评估。
# 例如run1：自动选该motion现有的最大iteration，做完整G1及随机phase扰动评估。
# 两项均达到19/20才更新当前入库名单；未合格保留原名单，报告留在artifacts/stage2/teacher_candidates。
# python scripts/stage2/workflow.py add-teacher --motion run1_subject2

# B. 全新motion：以下new_motion替换成真实动作名，先准备实际重定向CSV。
# 已有25Hz NPZ和teacher时，跳过转换/训练；teacher沿用仓库默认训练seed和num_envs。
# mkdir -p data/reproduction/raw_g1 data/reproduction/motions_25hz
# python scripts/csv_to_npz.py --input_file data/reproduction/raw_g1/new_motion.csv --input_fps 30 --output_fps 25 --output_file data/reproduction/motions_25hz/new_motion.npz --no_upload --exit_after_save --headless
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/new_motion.npz --run_name new_motion_default --headless --device cuda:0
# python scripts/stage2/workflow.py add-teacher --motion new_motion

# C. 新teacher入库后，采集该动作D0；原5项D0可复用。
# python scripts/stage2/workflow.py collect-d0 --motion new_motion
# 用所有当前入库动作的D0重新训练扩库版CVAE，重新估计完整D0训练集输入统计。
# 独立输出目录保留原5动作模型；数据变更不能当成原训练的精确断点恢复。
# python scripts/stage2/workflow.py train-d0 --output logs/stage2/cvae_d0_expanded
# python scripts/stage2/workflow.py evaluate --student logs/stage2/cvae_d0_expanded/best.pt
# 然后按第一类第4—6步，对全部当前入库motion做DAgger/评估；使用未占用的round编号。
# 替换已有motion的teacher或NPZ时，原D0身份会不匹配：用新的--data_root重新组织采集，勿覆盖旧数据。
# CVAE/decoder改变后，第三阶段latent数据需要重新采集，不能沿用旧decoder生成的z。

# 补充：workflow命令可加--dry_run，仅打印实际长命令，不启动仿真/训练或修改名单。
# 断点恢复和参数调整见docs/beyondmimic_reproduction/11_第二阶段代码与执行.md。
