# Teacher扩库与开训说明

2026-10-01。用户报告原4个teacher已在服务器训练；当前准备新增8个不同motion。显卡位置由用户分配，不假设4台服务器各3卡。每motion先训练一次，seed/num_envs使用仓库默认。

## 本轮实际完成

新增8个CSV下载自同一固定revision `ce1572906efe6157840e8474d5a0d7aa87481e74`，核对36列/30Hz/XYZW/29 joints/有限数值/四元数范数。实际调用Isaac FK转换器，全部生成25Hz NPZ且退出码0；复核29关节、30原生body、各字段连续长度与范数。当前官方URDF关节范围检查无越界条目（容差1e-3），这不是动力学可行性或teacher成功证据。

| motion_id | 类别（候选） | CSV时长s | NPZ帧数 | 开训顺序 |
|---|---|---:|---:|---|
| sprint1_subject2 | 冲刺候选 | 273.1 | 6828 | 首批扩库 |
| jumps1_subject1 | 跳跃候选 | 244.4 | 6111 | 首批扩库 |
| fight1_subject2 | 格斗候选 | 244.9 | 6122 | 首批扩库 |
| fightAndSports1_subject1 | 格斗与运动候选 | 245.1 | 6129 | 首批扩库 |
| dance2_subject1 | 另一组舞蹈候选 | 225.7 | 5642 | 首批扩库 |
| fallAndGetUp1_subject1 | 跌倒起身候选1 | 168.2 | 4205 | 先接触短训 |
| fallAndGetUp2_subject2 | 跌倒起身候选2 | 163.9 | 4098 | 先接触短训 |
| fallAndGetUp3_subject1 | 跌倒起身候选3 | 102.2 | 2555 | 先接触短训 |

类别来自文件名与运动学统计，用于选库，不保证每个片段包含某个准确技能，更不能宣称其中已有cartwheel。没有裁剪、抬高root、clip关节或修改原始CSV；因此源hash可追溯。当前数据并未补齐论文全部高动态能力。

## 现在怎样执行

本地迁移包：[teacher_expansion_8motions.tar.gz](../../artifacts/teacher_expansion_8motions.tar.gz)。包含新增8个CSV/NPZ、来源manifest、校验报告、audit工具、train.sh及本说明，不含原4个teacher的数据或权重。复制到目标服务器后，在项目根目录执行 `tar -xzf teacher_expansion_8motions.tar.gz`，再运行train.sh第5节的资产校验命令；该检查核对NPZ hash与当前URDF hash。包中包含更新的train.sh，解包前保留服务器上需要的自定义修改。

1. 同步本地 `data/reproduction/motions_25hz/` 中需要的新增NPZ到各目标服务器相同目录；复制对应原CSV、来源manifest及audit工具可复查完整资产。
2. 继续使用已能训练原4个teacher的环境/资产版本。原4个任务继续运行，本轮没有修改其reward/配置或启动重复任务。
3. 读取根目录[train.sh](../../train.sh)第5节。新增8条正式训练命令相互独立，全部注释，不设置seed/num_envs/max_iterations，不绑定物理GPU。由用户在终端或调度器分配空闲显卡；多任务不能未经分配都落到本机默认GPU0。
4. 先安排sprint/jumps/fight/fightAndSports/dance2。每条正式命令的run_name为`motion_id_default`，experiment_name为`g1_teacher_motion_id`，保存完整.pt与params/env.yaml、agent.yaml。
5. 三组fallAndGetUp先执行该小节的100轮probe和1env视频回放，检查reset/数值/接触异常，再决定正式训练。probe不要求100轮已经学会起身；40秒视频也不是完整技能验收。

## 起身动作的特殊检查

三组CSV最低root高度约0.04–0.12m，最大倾斜约84–142度。FK结果中最低body link origin约-0.054/-0.066/-0.062m。Link origin不是碰撞表面，因此这些数值只能提出地面接触检查，不能单凭它们判定真实碰撞几何穿透。

使用官方tracking终止：高度/朝向误差相对参考，而非“头低于固定高度就失败”。不要为起身动作临时加upright绝对高度终止，也不要用大范围关闭termination掩盖跟踪失败。检查数据轨迹与实际身体、接触、初始reference phase和reset后瞬间加速度；若某段确实不适配当前碰撞模型，保存失败phase/证据，单独裁选合法片段并用新的motion_id重新转换，不能无记录地改原数据。

接触动作不稳定时，先把已合格的非接触motion用于VAE首轮；其余motion后续合格后再加入，避免整体等待。任何新增teacher都需通过完整tracking验收，不能以CSV/FK通过取代teacher质量门槛。

## 证据和资产路径

- [新增CSV来源、revision、SHA256](references/expanded_raw_g1_manifest.json)
- [CSV/NPZ完整性与运动学审查](references/expanded_motion_asset_audit.json)
- [准备过程与命令验证](references/expansion_preparation_checks.json)：8次真实转换退出码0，校验器负例和8条独立命令语法通过。
- [12-motion当前teacher清单](configs/teacher_curriculum_v1.json)：前4项是用户报告training，新增8项是assets ready；没有任何权重被标为已验收。
- [audit_motion_assets.py](../../scripts/audit_motion_assets.py)：只依赖numpy的离线检查，不启动Isaac。使用方式见train.sh第5节。
- 实际转换日志：`logs/motion_preparation/motion_id.log`。该目录不提交git。

NPZ在 `data/reproduction/motions_25hz/`，CSV在 `data/reproduction/raw_g1/`；它们被gitignore排除，单独推送代码不会传输数据。新增文件本地已生成，服务器若复制完整数据就不必再下载或转换。所有文件hash写入审查报告。

下一步是逐motion记录teacher完整评估结果、锁定checkpoint+normalizer，然后实现并采集D0/DAgger。当前新增8个teacher未启动；尚没有合格teacher权重可用于正式VAE或扩散训练。
