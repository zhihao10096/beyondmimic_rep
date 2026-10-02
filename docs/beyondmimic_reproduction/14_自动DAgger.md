# 自动DAgger直到G2 clean

默认从`logs/stage2/cvae_d1/best.pt`开始，下一轮D2。脚本冻结启动时的入库teacher名单，顺序在同一可见GPU采集全部motion的train/val、聚合D0与历轮DAgger、训练并完整评估。只有全部motion通过G2 clean才正常结束；没有降低阈值或使用短程存活替代完整动作成功。

激活原teacher环境，将本次新增的`dagger.sh`、`scripts/stage2/auto_dagger.py`以及更新后的`scripts/stage2/workflow.py`同步到服务器，在仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash dagger.sh
```

默认不限制轮数。每轮使用32个环境采集train2500步、val1000步；CVAE训练50epochs、100000samples/epoch、batch512。可以传对应参数调整起始预算，但已有状态恢复时必须保持原预算、源码、初始权重和可见显卡配置一致。每轮执行固定预算后选该轮`best.pt`做实际闭环评估；自动迭代不保证能够收敛。

预览只展开一个循环，不启动物理仿真/训练，也不创建状态或假数据：

```bash
bash dagger.sh --dry_run
```

长时间运行可保存完整日志：

```bash
mkdir -p logs/stage2
CUDA_VISIBLE_DEVICES=0 nohup bash dagger.sh > logs/stage2/dagger_auto.log 2>&1 &
```

已有完整评估会先核对student SHA256、全部motion、teacher/GATE/配置/contract/mapping、执行源码、原始报告及计分。符合当前模型和协议的报告可以复用，包括失败报告；不依赖终端输出文字。你提供的D1模型SHA256为`3bd757d538ae70a6f42ae8953fe859beeff0d6bf98c0c5dca594773fef637bba`，本机对应原始评估已核对，五项0/20、G2未过。若服务器保留相同完整报告，默认自动找到并直接进入D2；找不到合法旧报告则先重新评估。

也可明确指定已有报告（必须同时保留其五份原始motion JSON）：

```bash
bash dagger.sh --initial_summary artifacts/stage2/student_evaluation/cvae_d1_20261002_132956_507819/summary.json
```

## 状态、恢复和退出

- 状态与冻结名单：`artifacts/stage2/auto_dagger/state.json`、`teacher_registry.json`；每轮验证过的评估汇总在其`evaluations/`下，原始物理报告仍保存在`artifacts/stage2/student_evaluation/`。状态记录当前checkpoint、SHA256、下一轮次、预算、源码和各阶段进度，同一状态目录不允许同时启动两个控制进程。
- 新数据：`data/stage2/动作名/d2_train_s102`、`d2_val_s202`，随后D3、D4……。旧D0/D1和完整历史轮次保留；默认训练输出`logs/stage2/cvae_d2`、`cvae_d3`……。训练与验证seed保持分离；第100轮起使用不重叠的新seed序列，避免无限循环时跨分区重复seed。
- Ctrl+C或SIGTERM会停止本任务的子进程。使用相同命令重新运行时，校验后复用完整采集、已完成评估和训练。若采集中断，仅把本任务已登记的不完整目录重命名为`*.interrupted_时间戳`保留原文件，再重新采集该split。训练中断从该轮`last.pt`按原参数恢复。不会自动删除外来数据或覆盖已有手动训练输出。
- 正常退出码0：`state.json`中`status=passed_g2_clean`，并打印合格模型和证据路径。退出码1表示命令失败、身份/数据不匹配或其他错误；130表示手动中断。这些都不称通过。
- 可用`--max_rounds 5`限制一次调用最多训练5轮；达到上限且未合格时退出码2、状态`limit_reached_not_passed`。再次运行继续已有状态。默认0表示不设轮数上限。
- 扩库或改变预算/源码时，应作为新实验处理，指定新`--state_dir`及必要的新`--model_root/--data_root`。冻结名单避免训练期间新teacher加入后悄悄改变验收目标；扩库仍按stage2.sh第二类执行。

通过后脚本停止。随后使用打印的最终checkpoint路径做完整扰动评估及decoder导出，不假定最终一定是D2，也不把clean G2通过当作扰动通过：

```bash
# 把最终checkpoint.pt替换成脚本打印的真实路径。
python scripts/stage2/workflow.py evaluate --student 最终checkpoint.pt --perturbed
python scripts/stage2/export.py --checkpoint 最终checkpoint.pt --output artifacts/stage2/decoder.pt
```

本次只验证了控制流程、命令展开、CPU数据/模型接口及旧真实报告，没有启动自动物理训练。控制器测试使用临时虚构模型/报告，不能作为真实G2证据；正式通过须来自服务器后续物理回放。
