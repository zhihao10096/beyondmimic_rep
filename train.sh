#!/usr/bin/env bash
## BeyondMimic：25Hz G1 teacher 训练命令，2026-10-01。
## 全部命令已加 # 注释，不会自动执行；按小节取消所需命令前的“# ”。
## 先完成0/1/2；第3节是单个teacher示例，第4节用于四台服务器，避免重复启动walk1。
## 激活你服务器上兼容的 Isaac Sim 4.5 / Isaac Lab 2.1 / RSL-RL 2.3.x 环境。
## 不指定或安装CUDA版本。CUDA_VISIBLE_DEVICES只选择显卡，不表示CUDA版本。
## 当前代码已补 --motion_file / --output_file / --no_upload，无需W&B账号。

## 0. 每个新终端先设置路径；PROJECT_ROOT替换为服务器实际项目目录。
# PROJECT_ROOT="/home/user/Desktop/whole_body_tracking"
# cd "$PROJECT_ROOT"
# mkdir -p data/reproduction/raw_g1 data/reproduction/motions_25hz logs/teacher_console
# nvidia-smi -L
# python -c 'import torch, isaaclab, isaaclab_rl, rsl_rl; print("torch:", torch.__version__); print("GPU count:", torch.cuda.device_count()); print("Isaac Lab:", isaaclab.__file__)'
# python -m pip install -e source/whole_body_tracking

## 如G1资源未安装，执行以下下载/解包；已安装则跳过。
# curl -fL --retry 3 -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
# tar -tzf unitree_description.tar.gz
# tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
# test -f source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf
# sha256sum unitree_description.tar.gz

## 1. 下载4个已审查的G1 CSV；有本地副本也可复制到raw_g1目录。
## 固定revision，保留数据集许可；CSV是参考动作，不是teacher action标签。
# mkdir -p data/reproduction/raw_g1 data/reproduction/motions_25hz logs/teacher_console
# DATA_REV="ce1572906efe6157840e8474d5a0d7aa87481e74"
# MOTIONS=(walk1_subject1 walk2_subject1 run1_subject2 dance1_subject1)
# for motion in "${MOTIONS[@]}"; do
#   curl -fL --retry 3 -o "data/reproduction/raw_g1/${motion}.csv" "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/${DATA_REV}/g1/${motion}.csv" || break
# done
# python - <<'PY'
# import hashlib, json
# from pathlib import Path
# manifest = json.loads(Path("docs/beyondmimic_reproduction/references/raw_g1_manifest.json").read_text())
# for entry in manifest:
#     path = Path("data/reproduction/raw_g1") / (entry["motion_id"] + ".csv")
#     assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Hash mismatch: {path}"
# print("4 CSV hashes matched")
# PY

## 顺序转换为25Hz NPZ。每个动作独立输出，保存后退出；失败停止。
## headless/no_upload/exit_after_save组合会验证并关闭输出后直接退出，绕过Kit清理卡住。
## 第一次转换也检查G1模型能否真正启动。此步骤使用GPU0，不要同卡同时跑teacher。
# MOTIONS=(walk1_subject1 walk2_subject1 run1_subject2 dance1_subject1)
# mkdir -p data/reproduction/motions_25hz
# for motion in "${MOTIONS[@]}"; do
#   CUDA_VISIBLE_DEVICES=0 python scripts/csv_to_npz.py --input_file "data/reproduction/raw_g1/${motion}.csv" --input_fps 30 --output_fps 25 --output_file "data/reproduction/motions_25hz/${motion}.npz" --no_upload --exit_after_save --headless --device cuda:0 || break
# done

## 2. GPU0单卡短训：先32环境/100轮，确认能启动、保存和恢复。
## 这是启动验证，不代表teacher已合格。低频任务必须搭配25Hz NPZ。
# CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/walk1_subject1.npz --num_envs 32 --max_iterations 100 --seed 0 --experiment_name g1_teacher_smoke --run_name walk1_subject1_seed0_smoke --logger tensorboard --headless --device cuda:0

## 3. 首个正式teacher：先只训练walk1，1024环境为profile起点。
## 无OOM、吞吐和物理表现正常后再考虑2048/4096；30000是训练预算，非通过标准。
## 用nohup在后台运行，标准输出独立保存；模型/配置在logs/rsl_rl/g1_teacher_walk1_subject1/。
# CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/walk1_subject1.npz --num_envs 1024 --max_iterations 30000 --seed 0 --experiment_name g1_teacher_walk1_subject1 --run_name walk1_subject1_seed0 --logger tensorboard --headless --device cuda:0 > logs/teacher_console/walk1_subject1_seed0.log 2>&1 &
# tail -f logs/teacher_console/walk1_subject1_seed0.log

## 回放最近一次walk1/seed0检查点：可视化需要显示环境；纯服务器用下面录视频命令。
## play会导出带normalizer的ONNX，需环境已安装项目的导出依赖。
# CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/walk1_subject1.npz --experiment_name g1_teacher_walk1_subject1 --load_run '.*walk1_subject1_seed0$' --checkpoint 'model_.*.pt' --num_envs 1 --device cuda:0
# CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/walk1_subject1.npz --experiment_name g1_teacher_walk1_subject1 --load_run '.*walk1_subject1_seed0$' --checkpoint 'model_.*.pt' --num_envs 1 --video --video_length 1000 --headless --device cuda:0
## 视频在该run的videos/play/。40秒回放只是诊断，不是完整动作/多次rollout验收。
## 本仓库仍有10秒episode timeout与reference末尾内部重采样；完整验收需按04文档补评估器。

## 如中断，恢复已有run；先确认load_run匹配唯一目标，必要时填完整run目录名。
## max_iterations是此次追加学习的轮数，不会自动扣除已训练轮数。
# CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/walk1_subject1.npz --num_envs 1024 --max_iterations 10000 --seed 0 --experiment_name g1_teacher_walk1_subject1 --run_name walk1_subject1_seed0_resume --resume True --load_run '.*walk1_subject1_seed0$' --checkpoint 'model_.*.pt' --logger tensorboard --headless --device cuda:0 > logs/teacher_console/walk1_subject1_seed0_resume.log 2>&1 &

## 4. 四台服务器分别训练：每台只选择下面对应的一条训练命令。
## 各自使用本机GPU0，不设置seed和num_envs，沿用仓库配置默认值。
## 当前任务默认4096环境、30000轮；seed继承RSL-RL配置，实际值见params/agent.yaml。
## 每台先进入项目根目录、激活环境，并复制对应的25Hz NPZ；无需MOTIONS等变量。
## 每台先执行下面的目录创建命令；日志与模型分别保存在本机logs目录。
## 如果walk1已经在训练或已合格，跳过服务器1命令。dance仍是候选动作。
# mkdir -p logs/teacher_console

## 服务器1：walk1_subject1
# cd /pfs/user/learning/whole_body_tracking
# source .venv/bin/activate
# python  scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/walk1_subject1.npz --experiment_name g1_teacher_walk1_subject1 --run_name walk1_subject1_default --logger tensorboard --headless

## 服务器2：walk2_subject1
# source .venv/bin/activate
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/walk2_subject1.npz --experiment_name g1_teacher_walk2_subject1 --run_name walk2_subject1_default --logger tensorboard --headless

## 服务器3：run1_subject2
# source .venv/bin/activate
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/run1_subject2.npz --experiment_name g1_teacher_run1_subject2 --run_name run1_subject2_default --logger tensorboard --headless

## 服务器4：dance1_subject1
# source .venv/bin/activate
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/dance1_subject1.npz --experiment_name g1_teacher_dance1_subject1 --run_name dance1_subject1_default --logger tensorboard --headless

## 本节run_name使用_default；回放/恢复时load_run应匹配本节名称，而非第3节的_seed0。
## 例如walk1：--load_run '.*walk1_subject1_default$' --checkpoint 'model_.*.pt'

## 后续阶段的12卡用途（这些模块尚未实现，没有伪造其训练CLI）：
## teacher扩库：12卡分别训练12个不同motion/片段，一个teacher对应一个动作。
## 每个motion先训练一次，按validation选择合格checkpoint；不稳定/失败的motion才追加训练seed。
## 同一teacher仍需不同评估seed/初始phase/扰动的多次rollout，这是评估，不是重新训练teacher。
## 当前4个teacher由用户报告正在训练；新增8个motion已在第5节准备，开训前分配空闲显卡。
## D0/DAgger：按motion/采集seed并行仿真采集，teacher提供标签，student执行DAgger。
## VAE：先单卡训练一个统一student，包含所有合格teacher，不需要每卡一套专属VAE。
## VAE+OU：12卡并行真实rollout，固定同一VAE权重，每卡写独立shard。
## diffusion：先4卡DDP，每卡128/global512，先训练一个模型；余卡可采集/评估/扩库。
## 3组×4卡不同训练seed仅在最终重复性评估时可选，不是当前必做任务。
## 最终12卡按任务/seed并行闭环评估。不要在teacher尚未合格时开始正式VAE/diffusion训练。

## 5. 新增8个不同motion：本地CSV和25Hz NPZ已准备；不重复启动前4个teacher。
## 由你决定每条命令在哪台服务器、哪张空闲卡运行，本节不指定CUDA_VISIBLE_DEVICES。
## 如一台机器并行多个job，请在各自终端/调度器先正确分配不同GPU；不要默认都占GPU0。
## seed/num_envs/max_iterations沿用仓库默认；每条训练命令独立、前台运行。
## 在目标服务器同步对应NPZ、当前脚本/文档后，从项目根目录激活环境。
# source .venv/bin/activate
# mkdir -p logs/teacher_console

## 资产校验：若复制了完整扩库包（包含CSV/NPZ/manifest/audit脚本），先运行以下命令。
# python scripts/audit_motion_assets.py --manifest docs/beyondmimic_reproduction/references/expanded_raw_g1_manifest.json --raw_dir data/reproduction/raw_g1 --npz_dir data/reproduction/motions_25hz --expected_report docs/beyondmimic_reproduction/references/expanded_motion_asset_audit.json --report logs/teacher_console/expanded_asset_audit.json

## 若没有复制本地数据，可在服务器下载；已经复制则跳过。下载只用CPU/网络。
# mkdir -p data/reproduction/raw_g1 data/reproduction/motions_25hz
# DATA_REV="ce1572906efe6157840e8474d5a0d7aa87481e74"
# EXTRA_MOTIONS=(sprint1_subject2 jumps1_subject1 fight1_subject2 fightAndSports1_subject1 dance2_subject1 fallAndGetUp1_subject1 fallAndGetUp2_subject2 fallAndGetUp3_subject1)
# for motion in "${EXTRA_MOTIONS[@]}"; do
#   curl -fL --retry 3 -o "data/reproduction/raw_g1/${motion}.csv" "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/${DATA_REV}/g1/${motion}.csv" || break
# done
# python scripts/audit_motion_assets.py --manifest docs/beyondmimic_reproduction/references/expanded_raw_g1_manifest.json --raw_dir data/reproduction/raw_g1 --report logs/teacher_console/expanded_csv_audit.json

## 如需重建NPZ，在分配好的空闲GPU上执行；本地8个NPZ已转换，无需重复转换。
# EXTRA_MOTIONS=(sprint1_subject2 jumps1_subject1 fight1_subject2 fightAndSports1_subject1 dance2_subject1 fallAndGetUp1_subject1 fallAndGetUp2_subject2 fallAndGetUp3_subject1)
# for motion in "${EXTRA_MOTIONS[@]}"; do
#   python scripts/csv_to_npz.py --input_file "data/reproduction/raw_g1/${motion}.csv" --input_fps 30 --output_fps 25 --output_file "data/reproduction/motions_25hz/${motion}.npz" --no_upload --exit_after_save --headless || break
# done

## 冲刺候选：sprint1_subject2
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/sprint1_subject2.npz --experiment_name g1_teacher_sprint1_subject2 --run_name sprint1_subject2_default --logger tensorboard --headless

## 跳跃候选：jumps1_subject1
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/jumps1_subject1.npz --experiment_name g1_teacher_jumps1_subject1 --run_name jumps1_subject1_default --logger tensorboard --headless

## 格斗候选：fight1_subject2
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fight1_subject2.npz --experiment_name g1_teacher_fight1_subject2 --run_name fight1_subject2_default --logger tensorboard --headless

## 格斗与运动候选：fightAndSports1_subject1
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fightAndSports1_subject1.npz --experiment_name g1_teacher_fightAndSports1_subject1 --run_name fightAndSports1_subject1_default --logger tensorboard --headless

## 另一组舞蹈候选：dance2_subject1
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/dance2_subject1.npz --experiment_name g1_teacher_dance2_subject1 --run_name dance2_subject1_default --logger tensorboard --headless

## 跌倒起身候选1：fallAndGetUp1_subject1
## 此动作有低姿态/低连杆原点，先短训与回放检查，成功后才启动下方正式训练。
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp1_subject1.npz --experiment_name g1_teacher_fallAndGetUp1_subject1_probe --run_name fallAndGetUp1_subject1_probe --max_iterations 100 --logger tensorboard --headless
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp1_subject1.npz --experiment_name g1_teacher_fallAndGetUp1_subject1_probe --load_run ".*fallAndGetUp1_subject1_probe$" --checkpoint "model_.*.pt" --num_envs 1 --video --video_length 1000 --headless
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp1_subject1.npz --experiment_name g1_teacher_fallAndGetUp1_subject1 --run_name fallAndGetUp1_subject1_default --logger tensorboard --headless

## 跌倒起身候选2：fallAndGetUp2_subject2
## 此动作有低姿态/低连杆原点，先短训与回放检查，成功后才启动下方正式训练。
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp2_subject2.npz --experiment_name g1_teacher_fallAndGetUp2_subject2_probe --run_name fallAndGetUp2_subject2_probe --max_iterations 100 --logger tensorboard --headless
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp2_subject2.npz --experiment_name g1_teacher_fallAndGetUp2_subject2_probe --load_run ".*fallAndGetUp2_subject2_probe$" --checkpoint "model_.*.pt" --num_envs 1 --video --video_length 1000 --headless
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp2_subject2.npz --experiment_name g1_teacher_fallAndGetUp2_subject2 --run_name fallAndGetUp2_subject2_default --logger tensorboard --headless

## 跌倒起身候选3：fallAndGetUp3_subject1
## 此动作有低姿态/低连杆原点，先短训与回放检查，成功后才启动下方正式训练。
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp3_subject1.npz --experiment_name g1_teacher_fallAndGetUp3_subject1_probe --run_name fallAndGetUp3_subject1_probe --max_iterations 100 --logger tensorboard --headless
# python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp3_subject1.npz --experiment_name g1_teacher_fallAndGetUp3_subject1_probe --load_run ".*fallAndGetUp3_subject1_probe$" --checkpoint "model_.*.pt" --num_envs 1 --video --video_length 1000 --headless
# python scripts/rsl_rl/train.py --task Tracking-Flat-G1-Low-Freq-v0 --motion_file data/reproduction/motions_25hz/fallAndGetUp3_subject1.npz --experiment_name g1_teacher_fallAndGetUp3_subject1 --run_name fallAndGetUp3_subject1_default --logger tensorboard --headless

## 100轮probe检查启动/reset/数值和接触问题，不要求此时已学会完整起身。
## 8个正式teacher命令均未执行；通过04文档的完整tracking验收后才能纳入VAE。
