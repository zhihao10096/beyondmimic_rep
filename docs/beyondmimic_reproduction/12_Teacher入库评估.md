# Teacher入库评估

日期：2026-10-02。用户提供 `logs/rsl_rl/g1_flat` 中11个checkpoint，选择每个训练目录现有iteration最大的文件并冻结SHA256。9项29999；fallAndGetUp1_subject1为20000，fallAndGetUp2_subject2为16500。fallAndGetUp3_subject1目前该目录下没有checkpoint，不计入此次11项。

## 预先确定的协议

- 完整G1：每motion20个并行trial，seed300、25Hz、phase0到motion最后一帧。保留startup材料/默认关节/COM随机化，关闭reset噪声和interval push，policy输入为clean命名观测。失败后不重新计分；motion末尾不teleport。沿用此前G1工程门槛：≥19/20成功。
- 鲁棒性补充：每motion20个trial，seed400，均匀随机phase，初始phase至少留有10秒reference。恢复原始reset pose/velocity/joint噪声和interval push，观测仍为clean，实际执行250个25Hz控制步=10秒。判定物理存活，与完整motion结束分别记录。该短程测试不能产生合格G1 gate。
- 推荐规则：完整G1通过且随机扰动存活≥19/20时推荐正式入库；仅完整G1通过则明确报告随机扰动不足，正式大规模D0之前先验证小批真实采集。完整G1未通过时不纳入整motion主库；若短程表现好，可以筛选片段、重新形成参考并重新验收，不能直接把失败帧删掉后称完整motion合格。
- 以上为工程筛选协议，不是论文作者公开的原始验收标准。20个trial是同一评估seed下不同startup/reset/phase随机抽样，不是重新训练20个teacher；报告Wilson95%区间和实际分母。

## 执行与证据

使用当前 `.venv` 在本机4070 Ti SUPER上顺序评估，避免同时启动多个Isaac进程挤占内存。不启动或改变服务器teacher训练；不使用pkl反序列化环境。actor和normalizer从真实checkpoint及原agent.yaml读取，env.yaml用数据加载器核对25Hz、观测、控制器、机器人配置和随机化事件。

批量入口：`scripts/stage2/evaluate_teachers.py`。命令：

```bash
python scripts/stage2/evaluate_teachers.py \
  --root logs/rsl_rl/g1_flat \
  --output artifacts/teacher_evaluation/2026-10-02_0946 \
  --num_envs 20
```

已存在输出目录时，只有显式`--resume`才能继续；仅当checkpoint、agent/env、motion、执行源码、评估seed和环境数的signature完全一致时复用完整报告。每个child独立保存JSON、日志和SHA256；加载/执行错误标为error，不计成0%物理失败。

实时表格：`artifacts/teacher_evaluation/2026-10-02_0946/summary.md`；结构化总表和全命令：同目录`summary.json`。评估结束后生成`references/teacher_evaluation_2026-10-02.json`和`13_Teacher评估结果_2026-10-02.md`作为可版本化证据。

合格motion的`*_full_clean.json`可以直接作为`stage2.sh`中的`GATE`用于正式D0。同一个gate绑定具体checkpoint及motion，换更晚checkpoint或改参考文件必须重验。此次不自动启动D0或CVAE训练。
