# v9：完整可行集监督、更密的后果监督、底座换提示词（一条命令全部跑完）

审稿意见反复追问两点：每个场景 120 个动作的结果都有，训练却只用了一个标签或少数几个候选；未训练底座 574/600 次复制了提示里的零动作示例。v9 在主划分上一次回答这两点。

## 实验

**训练**（底座 Qwen3.5-35B-A3B，协议与 v2 完全相同：模型、LoRA、优化器、有效批量 64、微批 1、损失、检查点、贪心解码，`tests/test_v9.py` 检查；各 3 个种子，共 6 次）

| 组 | 教什么 | 对照 |
|---|---|---|
| `A_set` | 与 A_base 相同的 15,000 次动作曝光、同样的场景顺序；每个答案先列出该题**整个**带余量的可行集（`{"feasible_moves": [[right, up, back], ...]}`，排序后），再给出 A_base 的动作。测试题面不变，打分器读最后一个 JSON 里的动作 | v2 A_base、A_compute；v8 A_multi |
| `A_cons_dense` | A_base 的 15,000 次动作曝光，加上全部 111,634 条 consequence_v9 记录各一次：每个训练场景最多 24 个候选（8 个满足、4 个终点不合法、12 个不满足），格式与 consequence_v4 相同，consequence_v4 是它的子集。是 v8 A_cons_full 的 3.8 倍、v4 A_cons 的 20.6 倍 | v4 A_cons、v8 A_cons_full、v2 A_base |

**只推理**（不训练，不加适配器；Qwen3.5-35B-A3B 和 Gemma 4 31B 两个未训练底座，各 600 道测试题）

| 条件 | 题面 | 设置 |
|---|---|---|
| 随机示例 | 提示末尾的输出示例 `{"move_m": {"right": 0, "up": 0, "back": 0}}` 换成一个非零动作，每题按 `sha256(task_id|v9|example)` 从 119 个非零动作里取 | 关闭 thinking，512 token |
| 只描述格式 | 示例换成"键 move_m 下给出三个整数 right、up、back，取值见上面的动作约定"，不给示例值 | 关闭 thinking，512 token |
| 打开 thinking | 原题面不变 | 打开 thinking，8192 token；拆成 60 份、每份 10 题，分到所有 GPU |

## 数据

增量包 `locus-bundle-v1.7.0-addon-v9.tar.gz`，由基准方单独传过去，sha256 在服务器说明里。里面有：
- `training_exports/train/consequence_v9.jsonl`
- `eval_prompts/test_V0_example_random.jsonl`、`eval_prompts/test_V0_schema.jsonl`
- `eval_prompts/v9_think/test_V0_part-00 … 59.jsonl`
- `ADDON_V9_MANIFEST.json`

v7 增量包已在数据包里（v8 用过），探针要用到。

## 一条命令跑完

```bash
export LOCUS_ROOT=/共享盘/项目目录
```

```bash
NODES=$LOCUS_ROOT/work/v9/nodes.txt ADDON_TAR=$LOCUS_ROOT/data/locus-bundle-v1.7.0-addon-v9.tar.gz ADDON_SHA256=<服务器说明里的 sha256> nohup scripts/run_v9.sh > /dev/null 2>&1 &
```

- `NODES`：每行一个节点 IP，运行命令的节点也要写进去。
- Gemma 权重默认在 `$LOCUS_ROOT/models/gemma-4-31B-it`，不在那里就加 `GEMMA_MODEL=...`。虚拟环境用 v7/v8 那个，要能同时加载 Qwen 和 Gemma。
- 步骤依次是：
  1. 核对环境、增量包和文件；
  2. 单元测试；
  3. 用真实处理器检查两种训练数据；
  4. 训练 6 次（每个节点一次跑一个，A_cons_dense 先跑）；
  5. 验证集选检查点；
  6. v2 整套测试题（每次训练和 B0；B0 与 v2 对照，核对推理环境没变）；
  7. C1/C2；
  8. v7 探针（S 题、改名题，A_cons_dense 加后果探针）；
  9. 底座换提示词（Qwen 用 v9 配置，Gemma 用 v3_gemma 配置）；
  10. 打包。
- 时间：A_set 每次约 30 分钟，A_cons_dense 每次约 3–4.5 小时（v8 A_cons_full 的 2.9 倍曝光）。底座推理约 1 小时。8 个节点总共约 8–10 小时。
- 中途停下，修好原因后运行同一条命令，已完成的步骤会跳过。
- 进度看 `$LOCUS_ROOT/work/v9/run_v9.log`。

## 打分（运行前写定）

区间按场景重抽样（先对 3 个种子取平均），95% 百分位。只报数值和区间，不设"有效"门槛。

1. **A_set**：
   - 测试集留出族成功率对 A_base、A_compute、A_multi 的差值；同时报告已见族、留出组合、F13、总分，以及已见族减留出族的差距。
   - 它列出的可行集：按划分报告对真实可行集的精确率、召回率，以及动作落在自己所列集合里的比例。
2. **A_cons_dense**：
   - 留出族成功率对 A_cons_full、A_cons、A_base 的差值。
   - 后果曝光量 5,416 / 29,400 / 111,634 三个点（A_cons / A_cons_full / A_cons_dense）的留出族成功率与后果探针：验证集、面板三个划分的平衡准确率，包括单候选格式和 12 候选格式。
3. **两个底座 × 三种提示**：按划分报告成功率、零动作比例、照抄示例的比例（随机示例条件）、格式错误、没写完的比例（thinking），并与原题面 B0（Qwen 0.8%、Gemma 7.5%）对照。没写完的回答按失败计。
4. **其余与 v8 相同**：S 题、改名题、C1/C2。

## 发回

跑完把 `work/v9/locus-v9-results-<日期>.tar` 和同名 `.sha256` 发回基准方打分。
