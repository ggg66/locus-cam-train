# v7：只推理，不训练（已有检查点上的诊断）

v7 不训练任何模型，只在 v2、v3_gemma、v4、v6、v6_gemma 已经训练好的终点检查点（ckpt-4）上做推理。回答五个问题：

| 问题 | 做法 | 用到的检查点 | 题目 |
|---|---|---|---|
| 同一个微调模型拿到真实几何后，能不能选对动作 | 题面写入参考几何（`test_diagnostic_S`，和 API 模型用的完全相同） | 全部 36 个微调检查点，加两个未训练底座 | 120 道诊断题 |
| 同上，但用模型熟悉的格式 | Geometry→Action 的答案开头直接给定几何，让模型接着写动作：真实几何、左右镜像后的几何各一遍 | v4 的 A_geomact，3 个 | 600 道测试题 × 2 |
| 完整的几何路线 | 模型写出场景几何，基准方用固定求解器算动作 | Qwen 和 Gemma 的 Action+Geometry，6 个 | 600 道测试题 |
| 留出族里训练没见过的物体名是不是差距的一部分 | 留出族题面里的 4 个新物体名换成训练里的名字（屏风、隔墙→隔断，吧台→展架，高植物→立柱），图片不变 | 全部 36 个微调检查点 | 200 道留出族题 |
| 辅助的后果任务有没有学会、能不能迁移 | 用 Action+Consequence 训练时的单候选格式，判断 12 候选面板里的每个动作；另判断验证集（已见族）上的同格式题 | v4 的 A_cons，3 个 | 1,440 道面板题 + 2,330 道验证题 |

推理设置和之前各轮相同：批量 1，贪心解码，关闭 thinking。生成上限：S 题和改名题 512（A_geomact、A_imagine、A_propose 用 1024），几何 1024，接着写动作 128，后果题 512。

## 数据

增量包 `locus-bundle-v1.7.0-addon-v7.tar`，sha256 `8f7c617abef177d42a48c66edb7fb115e20f91f6f5d784b0d846b112cb385669`，由基准方单独传过去。里面只有提示文件：

- `eval_prompts/test_geom_predict.jsonl`：600 道测试题的几何提示词，和 `test_diagnostic_geom_predict` 在那 120 题上逐字相同；
- `eval_prompts/test_geomact_true.jsonl`、`test_geomact_mirror.jsonl`：600 道测试题，`assistant_prefix` 是真实场景几何（Geometry→Action 训练答案的格式，后接换行），或它的左右镜像；
- `eval_prompts/test_V0_renamed_heldout.jsonl`：200 道留出族题，只改了题面里的物体名；
- `eval_prompts/test_cons_panel.jsonl`：120 道诊断题 × 12 个面板候选，Action+Consequence 训练格式；
- `eval_prompts/validation_cons.jsonl`：验证集 2,330 道同格式题。

`scripts/infer.py` 遇到 `assistant_prefix` 时把它放在助手回答的开头，模型接着写；输出的 `raw` 只含续写部分。`test_diagnostic_S` 在原数据包里。

## 一条命令跑完

```bash
LOCUS_ROOT=/共享盘/项目目录 NODES=$LOCUS_ROOT/work/v7/nodes.txt ADDON_TAR=$LOCUS_ROOT/data/locus-bundle-v1.7.0-addon-v7.tar ADDON_SHA256=8f7c617abef177d42a48c66edb7fb115e20f91f6f5d784b0d846b112cb385669 nohup scripts/run_v7.sh > /dev/null 2>&1 &
```

- `NODES`：每行一个节点 IP，运行命令的节点也要写进去。推理分到所有节点的所有 GPU。
- 步骤：核对增量包和文件、v7 单元测试；按轮次生成任务并推理（Qwen 的 v2、v4、v6，然后 Gemma 的 v3_gemma、v6_gemma）；打包。
- 缺终点检查点的运行会跳过，写进 `work/v7/missing.txt`，最后在日志里列出。
- 规模：约 4,600 条 S 题、3,600 条几何、3,600 条续写、7,200 条改名题、11,300 条后果题。8 个节点约一天。
- 中途停下，修好原因后运行同一条命令，已完成的步骤和答案会跳过。
- 进度看 `$LOCUS_ROOT/work/v7/run_v7.log`。

## 打分（运行前写定，基准方用冻结评分器）

都按测试划分（已见 / 留出族 / 留出组合）分开报告，按场景重抽样给 95% 区间：

1. **真实几何（题面）**：每个检查点在 120 题上给几何与不给几何（已有 V0 答案）的成功率之差。
2. **真实几何（答案开头）**：A_geomact 接真实几何后的成功率，对比它自己写几何时的 V0 成功率和求解器从真实几何出发的成功率；动作落在所给几何的求解器可行集内的比例。镜像几何：动作跟着镜像几何走（在镜像几何的可行集内）的比例，对比跟着图片走（在真实可行集内）的比例。
3. **完整几何路线**：AG 的几何接求解器，600 题成功率，对比 AG 自己直接给的动作。
4. **改名**：同一检查点在 200 道留出族题上改名前后的成功率之差（配对）。
5. **后果探针**：A_cons 在验证集（已见族）和面板（三个划分）上大小、遮挡标签的平衡准确率，以及高度占比的数值误差。

## 发回

跑完把 `work/v7/locus-v7-results-<日期>.tar` 和同名 `.sha256` 发回基准方打分。包里是全部答案、任务清单、`missing.txt` 和日志，不含权重。
