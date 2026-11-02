# v6：最终测试（先想象再挪动）

v5 在开发集上比较了四种方法，“提出候选、想象结果、再选择”的微调阶段（v5 的 `m2_imagact`）最好：没训练过的两个族上从 2.94 提高到 29.85，三个种子都在 27.7–32.6。v6 用全部训练族重新训练这个方法，在冻结测试集上只测一次。

结论需要的三件事在同一次运行里完成，跑完不需要再补实验：

| 问题 | 实验 | 底座 |
|---|---|---|
| 方法有没有效 | `A_imagine` 对比已有的 A_base、A_compute | Qwen3.5-35B-A3B（`configs/v6/`） |
| 效果是不是来自想象的结果 | `A_propose`：同样的候选和选择，不写想象的结果 | Qwen3.5-35B-A3B |
| 换一个底座还成不成立 | `A_imagine` 对比已有的 Gemma A_base、A_compute | Gemma 4 31B（`configs/v6_gemma/`） |

每组 3 个种子，共 9 次训练。

## 训练

- `A_imagine`：答案先列 4 个候选挪法，各写出移动后的结果，再选出结果全部满足的那个。和 v5 的 `m2_imagact` 完全相同，只是不再留出 F09、F14；不接强化学习。
- `A_propose`：同一批记录，4 个候选挪法和顺序、选中的挪法都不变，只去掉每个候选的结果。数据 `propose_v6` 由 `scripts/make_propose_v6.py` 在服务器上从 `imagact_v5` 生成，sha256 与基准方生成的一致才继续。
- 协议与已有对照组相同：Qwen 沿用 v2，Gemma 沿用 v3_gemma（模型、LoRA、优化器、有效批量 64、微批 1、损失、检查点、贪心解码都不变，`tests/test_v6.py` 检查）。15000 次曝光，场景顺序和同种子的 A_base 完全一致（setup 用真实数据核对）。
- 对照组用已经跑完的 A_base（同样 15000 次动作）和 A_compute（同训练量、只学动作），不重训。
- 预设评测点是终点检查点（ckpt-4）；验证集选出的最优检查点另外报告，和 v2 一样。

## 判定标准（运行前写定）

主指标是测试集留出结构（test_geometry）上的成功率。

1. **方法有效**（Qwen 的 `A_imagine`，沿用 v4 的标准）：比 A_compute、比 A_base 都至少高 5 个点，且按场景重抽样的 95% 区间下限大于 0；3 个种子里至少 2 个同时高于这两个对照组的均值。
2. **换底座成立**（Gemma 的 `A_imagine`）：同一条标准，对照组是 Gemma 的 A_base、A_compute。
3. **效果来自想象的结果**（Qwen）：`A_imagine` 比 `A_propose` 至少高 5 个点，且 95% 区间下限大于 0。

同时报告分布内（test_id）、组合（test_composition）、F13、总分、几何路线、后果面板、C1、C2。打分由基准方用冻结评分器完成（`score_round.py v6`、`score_round.py v6_gemma`），测试结果出来后不再改任何训练或评测设置。

## 运行

服务器上沿用 v5 的数据包和环境，不需要新的增量包；Gemma 权重用 v3_gemma 那一轮下载的（默认 `$LOCUS_ROOT/models/gemma-4-31B-it`，别处请设 `GEMMA_MODEL`）。先更新代码（v5 留在服务器上的本地补丁不涉及本次改动的文件，可以直接更新），然后：

```bash
LOCUS_ROOT=/共享盘/项目目录 NODES=$LOCUS_ROOT/work/v6/nodes.txt AIHUB_API_KEY=... nohup scripts/run_v6.sh > /dev/null 2>&1 &
```

- `NODES`：每行一个节点 IP，运行命令的节点也要写进去。每个节点一次跑一个训练，9 次训练按节点数分批；推理分到所有节点。
- `AIHUB_API_KEY`：必须设置，一线模型诊断（`scripts/run_api_diagnostics.py`）在后台同时跑，结果一起打包；已经跑完（存在 `work/api_diag.tar.gz`）就跳过。没有 key 又没跑完时脚本会停下，确实不跑才设 `SKIP_API=1`。
- 步骤：核对文件和数据对齐、生成 `propose_v6`、单元测试、用两个底座的真实处理器检查全部训练数据、训练、验证集选检查点、测试（每次训练和两个底座的 B0 各跑一遍 v2 的整套测试题；B0 与之前同底座的 B0 对照，核对推理环境没有变）、C1/C2、打包。推理批量 1，和对照组相同。
- 时间：Qwen 每次训练约 30 分钟，Gemma 约 2 小时；推理约 4 万条。8 个节点大约一天内跑完。
- 中途停下，修好原因后运行同一条命令，已完成的步骤会跳过。
- 进度看 `$LOCUS_ROOT/work/v6/run_v6.log`，训练日志在 `work/logs/<v6 或 v6_gemma>-<组>-s<种子>.log`。

## 发回

跑完把 `work/v6/locus-v6-results-<日期>.tar` 和同名 `.sha256` 发回基准方打分。这一个包里有：

- 两个底座的全部测试、验证、C1/C2 答案和训练记录，检查点的 sha256（不含权重，检查点先保留在服务器上）；
- v5 的原始答案、训练记录、`runtime_fixes/`，以及代码目录里未提交的改动（`code_diff.patch`、`code_untracked.tar.gz`）；
- 一线模型诊断的结果包。
