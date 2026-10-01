# 参考材料与来源锁

本目录用于证据核对，不是已训练的数据或模型目录。

- [sources_lock.json](sources_lock.json)：官方与两个第三方checkout的commit，以及下载参考文件的字节数和SHA256。数据固定revision另见[raw_g1_manifest.json](raw_g1_manifest.json)。
- [shared_conversation.md](shared_conversation.md)：用户分享链接的可见对话正文。
- `paper_v4.html`、`paper_v4_text.txt`：论文v4正文与补充材料的离线快照。论文来源：https://arxiv.org/html/2508.08241v4 。
- `repos/BeyondMimic-Reproduction/`、`repos/UniPhys/`：只读源码参考。不要在这里实现本项目；复用内容时保留原许可，并记录源commit/文件。
- `raw_g1/`：4个G1参考CSV，附数据集README和许可材料。不是teacher动作标签，也不是真实state–latent rollout。
- `official_issue6_comments.json`、`official_repo_metadata.json`、`zenodo_record.json`、`lafan_dataset_metadata.json`、`robot_asset_headers.json`：当次来源核验记录；不是实时服务状态。
- [offline_probes.json](offline_probes.json)：有限离线数值探测，不代表物理闭环通过。
- [preparation_qa.json](preparation_qa.json)：文档链接、JSON/YAML解析、CSV hash和原源码未修改的最终检查。
- [teacher_command_checks.json](teacher_command_checks.json)：后续训练入口补丁的CLI/帧率检查/语法验证，未启动IsaacSim。原preparation_qa是准备阶段的历史快照。

论文快照、原始CSV和外部checkout被本准备包`.gitignore`排除。当前文件保存在本地；仅执行git提交不会携带全部快照。迁移时复制整个准备包，或按锁定来源重新取得文件，再校验hash。

这些材料可能包含外部版权内容；此处供本项目研究核对，不能把整个参考目录作为自己编写的代码或数据公开发布。官方机器人资源目前只有可达性记录，T00才下载、校验和安装。Zenodo大ZIP没有下载。
