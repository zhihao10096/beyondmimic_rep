# 设计模板，尚未接入执行入口

这些文件供T00–T08实现配置加载器时使用，**不能假定现在的官方train.py支持它们**。字段名称是拟议schema，实施者可以调整结构，必须保持02契约和记录schema版本。

- `paper_v4_design.yaml`：P4给定值与工程选择分别成段。
- `resource_profiles.yaml`：本地/单卡/4卡配置起点；profile后选用。
- `teacher_map.template.json`：每motion权重/normalizer/motion/env的可迁移关联；所有status都是待训练，路径为空，禁止当可用teacher读取。

所有维度、frame、时间与normalizer定义都应在加载时验证。checkpoint复制resolved config及hash，不能只保存模板名称。`null`表示尚未确定，不代表默认0。
