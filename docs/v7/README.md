# v7：只推理，不训练（已有检查点上的三项诊断）

v7 不训练任何模型，只在 v2、v3_gemma、v4、v6、v6_gemma 已经训练好的终点检查点（ckpt-4）上做推理，回答三个问题：

| 问题 | 做法 | 用到的检查点 | 题目 |
|---|---|---|---|
| 同一个微调模型拿到真实几何后，能不能选对动作 | 题面写入参考几何（`test_diagnostic_S`，和 API 模型用的完全相同） | 全部 36 个微调检查点，加两个未训练底座 | 120 道诊断题 |
| 完整的几何路线 | 模型写出场景几何，基准方用固定求解器算动作 | Qwen 和 Gemma 的 Action+Geometry（AG），共 6 个 | 全部 600 道测试题 |
| 动作是否跟着答案里写的几何走 | Geometry→Action 的答案开头直接给定几何，让模型接着写动作：一种是真实几何，一种是左右镜像后的几何 | v4 的 A_geomact，3 个 | 全部 600 道测试题，各两种 |

推理设置和之前各轮相同：批量 1，贪心解码，关闭 thinking。生成上限：S 题 512（A_geomact、A_imagine、A_propose 用 1024），几何 1024，接着写动作 128。

## 数据

增量包 `locus-bundle-v1.7.0-addon-v7.tar`，sha256 `3f8e19b0a218cf1334b5a955aed4b42549c0e28d3c2bafb4b8402dc9f450800c`，由基准方单独传过去。里面只有三个提示文件：

- `eval_prompts/test_geom_predict.jsonl`：600 道测试题的几何提示词，和 `test_diagnostic_geom_predict` 在那 120 题上逐字相同；
- `eval_prompts/test_geomact_true.jsonl`：600 道测试题，`assistant_prefix` 是真实场景几何（Geometry→Action 训练答案的格式）加一个换行；
- `eval_prompts/test_geomact_mirror.jsonl`：同样的几何左右镜像（两侧墙对调，所有 right 坐标取反）。

`scripts/infer.py` 遇到 `assistant_prefix` 时把它放在助手回答的开头，模型接着写；输出的 `raw` 只含续写部分。`test_diagnostic_S` 在原数据包里。

## 一条命令跑完

```bash
LOCUS_ROOT=/共享盘/项目目录 NODES=$LOCUS_ROOT/work/v7/nodes.txt ADDON_TAR=$LOCUS_ROOT/data/locus-bundle-v1.7.0-addon-v7.tar ADDON_SHA256=3f8e19b0a218cf1334b5a955aed4b42549c0e28d3c2bafb4b8402dc9f450800c nohup scripts/run_v7.sh > /dev/null 2>&1 &
```

- `NODES`：每行一个节点 IP，运行命令的节点也要写进去。推理分到所有节点的所有 GPU。
- 步骤：核对增量包和文件、v7 单元测试；按轮次生成任务并推理（Qwen 的 v2、v4、v6，然后 Gemma 的 v3_gemma、v6_gemma）；打包。
- 缺终点检查点的运行会跳过，并写进 `work/v7/missing.txt`，最后在日志里列出。
- 规模：约 4,600 条 S 题答案、3,600 条几何、3,600 条续写。8 个节点大约半天。
- 中途停下，修好原因后运行同一条命令，已完成的步骤和答案会跳过。
- 进度看 `$LOCUS_ROOT/work/v7/run_v7.log`。

## 发回

跑完把 `work/v7/locus-v7-results-<日期>.tar` 和同名 `.sha256` 发回基准方打分。包里是全部答案、任务清单、`missing.txt` 和日志，不含权重。
