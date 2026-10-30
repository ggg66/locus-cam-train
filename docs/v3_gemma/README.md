# v3_gemma：换第二个底座（Gemma 4 31B）的服务器操作说明

这一轮把第二轮（v2）的三组训练原样搬到第二个底座上，用来检验 v2 的结论是不是只属于 Qwen3.5-35B-A3B。

| 组 | 训练什么 | 曝光次数 |
|---|---|---|
| A_base | 只学动作 | 动作 15000 |
| AG | 动作加几何 | 动作 15000（同种子与 A_base 顺序相同）+ 几何 5000 |
| A_compute | 只学动作，训练计算量与 AG 相同 | 动作次数由本轮重新标定 |

B0（未训练的 Gemma 4 31B）只做推理，同时作为主榜上的一个零样本开源模型。

## 一条命令跑完全部实验

```bash
export LOCUS_ROOT=/共享盘/项目目录
```

```bash
NODES=$LOCUS_ROOT/work/v3_gemma/nodes.txt nohup scripts/run_all_v3.sh > /dev/null 2>&1 &
```

`nodes.txt` 每行写一个节点 IP（8 个节点），运行这条命令的节点必须在其中。进度写在 `$LOCUS_ROOT/work/v3_gemma/run_all.log`，训练进度看 `LCAM_ROUND=v3_gemma launch/status.sh`。

脚本按顺序执行下面各步。每一步的检查通过后，会在 `work/v3_gemma/stages/` 留一个标记：

| 步骤 | 做什么 | 不通过就停下的条件 |
|---|---|---|
| setup | 检查 transformers 能否加载 Gemma 4；下载固定修订号的权重；检查数据包里有 v2 和 10-26 两个增量包 | 版本不对、模型不对、文件缺失 |
| unit | 单元测试；在服务器上重建 C2 提示词，与冻结版本逐字节核对 | 任一测试失败 |
| smoke | 缩小版随机 Gemma 4 走完全部脚本（单卡和 2 卡） | 没有输出 smoke passed |
| check / stats | 真实处理器核对训练集和验证集；统计每条样本的 token 数 | 有编码错误 |
| audit | 损失审计和批次追踪 | 学到的不是"答案 + `<turn|>`"，或图片数不对 |
| equiv | 单卡和 8 卡第一步梯度对比 | 梯度不一致；LoRA 不是只挂在注意力层和 MLP 上 |
| batchcheck | 推理批量 1 和 8 的答案对比 | 不停下：答案不一致时，后面的推理自动改用批量 1 |
| short | 12 步短训练，测每步用时，重载适配器核对 | 训练出错或重载不一致 |
| freeze | 按 token 数标定 A_compute 并冻结 | 超出容差 |
| train | 9 次训练按节点排队，每个节点同时只跑一次 | 同一次训练失败两次（第一次失败会自动重跑） |
| val / test | 验证集选检查点；终点和 B0 跑整套评测题 | 推理失败；test_V0 答案不是 6000 条 |
| confirm | 用每次训练自己的 V0 答案生成 C2 提示词；C1（B0 + 9 个终点）和 C2（9 个终点）推理 | 推理失败 |
| pack | 打包全部结果和 sha256（不含适配器权重） | — |

中途停下时，修好原因再运行同一条命令，已完成的步骤会跳过。要重做某几步：`REDO="val test" NODES=... scripts/run_all_v3.sh`。

跑完后，把 `work/v3_gemma/locus-v3_gemma-results-<日期>.tar` 和同名 `.sha256` 发回基准方。打分全部在基准方完成。

下面第一到第四部分是逐步手动执行时的说明，和上表的步骤一一对应。

## 和 v2 的区别

只改了三处，其余设置（数据、曝光次数、优化器、有效批量 64、微批 1、损失、检查点、贪心解码、评测题）与 v2 完全一样，配置见 `configs/v3_gemma/`。

1. **底座**：`google/gemma-4-31B-it`，修订号 `842da3794eaa0b77d5f08bae87a17459d91ff475`。这是 60 层的稠密模型，v2 的底座是 MoE。
2. **对话结束标记**：Gemma 4 的模板用 `<turn|>`（id 106）结束一轮回答，v2 是 `<|im_end|>`。关闭思考时，模板会先放一个空的 thought 通道，作用和 Qwen 的空 think 块一样。训练和推理用的前缀完全相同。
3. **LoRA 位置**：秩和缩放与 v2 相同（r 64，alpha 128，dropout 0.05），挂在注意力层和稠密 MLP 上，对应 v2 的"注意力 + 共享专家"。视觉塔、多模态投影层、词嵌入和 lm_head 都冻结。

**A_compute 必须重新标定。** Gemma 的分词和图像 token 数与 Qwen 不同：每张图约 260 个视觉 token，一条动作样本约 1030 个 token。所以 A_compute 的动作次数要在本轮重新按处理的 token 数算，不能沿用 v2 的 21696。

## 一、准备

1. **更新代码。** 拉取本仓库的 `main` 分支。v2 的默认行为没有改，`LCAM_ROUND` 不设置时仍然是 v2。
2. **下载权重。** 模型不需要申请访问权限：

```bash
"$LOCUS_ROOT/venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31B-it', revision='842da3794eaa0b77d5f08bae87a17459d91ff475', local_dir='$LOCUS_ROOT/models/gemma-4-31B-it')"
```

3. **确认 transformers 版本支持 Gemma 4。** 本地测试用的是 transformers 5.22.0、peft 0.21.2、torch 2.15.0。服务器上 v2 用的 5.20 理论上也支持，以 `unit` 和 `smoke` 两个阶段能否通过为准。
4. **设置环境变量。** 之后所有命令都要带上这一项：

```bash
export LOCUS_ROOT=/共享盘/项目目录 LCAM_ROUND=v3_gemma
```

设置后，`LCAM_MODEL` 默认指向 `$LOCUS_ROOT/models/gemma-4-31B-it`。配置读 `configs/v3_gemma/`，输出写到 `work/v3_gemma/`、`work/runs/v3_gemma/` 和 `work/outputs/v3_gemma/`，不会覆盖 v2 的结果。分发到其他节点的训练和推理命令会自动带上这两个变量。

数据包和 v2 共用，不需要新的增量包，`addon` 这一步跳过。

## 二、训练前的门禁（按顺序跑，结果在 `$LOCUS_ROOT/work/v3_gemma/audit/`）

```bash
SMOKE_SNAPSHOT=$LOCUS_ROOT/models/gemma-4-31B-it scripts/run_v2.sh unit smoke check stats audit
```

```bash
scripts/run_v2.sh equiv batchcheck short calibrate
```

各阶段的含义和 v2 相同，见 [v2 说明](../v2/README.md) 第二部分。本轮要特别看三处：

- `smoke`：用缩小版随机 Gemma 4（真实分词器、模板和图像处理器）把每个脚本走一遍。本地已在 CPU 上走通编码、LoRA 挂载、前向和反向、单条与批量推理。
- `audit`：
  - `batch_trace` 里每条样本学到的必须正好是"答案 + `<turn|>`"；
  - `two_images` 检查通过；
  - `run.json` 里 LoRA 只挂在 `attention` 和 `mlp` 两类层上。
- `short`：记下每步秒数和显存峰值。31B 稠密模型每个 token 的计算量约是 v2（每 token 激活约 3B 参数）的 10 倍，单次训练会比 v2 慢很多。9 次训练分到 8 个节点上，需要两批跑完。

把 `work/v3_gemma/audit/` 整个目录发回基准方，逐项核对后再冻结标定：

```bash
scripts/run_v2.sh freeze
```

冻结后，把 `configs/v3_gemma/calibration.json` 提交进仓库。

## 三、正式训练（需要授权）

```bash
ALLOW_TRAIN=1 NODES=$LOCUS_ROOT/work/v3_gemma/nodes.txt scripts/run_v2.sh train
```

```bash
launch/status.sh
```

## 四、评测与打包

```bash
NODES=$LOCUS_ROOT/work/v3_gemma/nodes.txt scripts/run_v2.sh val test pack
```

评测题、B0 对照和打包内容都与 v2 相同，结果包名为 `locus-v3_gemma-results-<日期>.tar`。打分在基准方完成。

## 五、本轮不做的事

不调超参数、不改 LoRA 位置和秩、不解冻视觉塔，也不会因为哪一组结果好就追加训练。如果门禁发现必须改的地方（例如显存放不下），先停下，把审计结果发回基准方，再一起决定。
