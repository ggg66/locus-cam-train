# locus-cam-train

> **国际一线模型零样本评测（只推理）**：`scripts/run_api_models.py`，模型清单、费用和用法见 [docs/api_frontier/README.md](docs/api_frontier/README.md)。
>
> **v3_gemma（第二个底座 Gemma 4 31B，同一套 v2 协议）**：操作说明见 [docs/v3_gemma/README.md](docs/v3_gemma/README.md)，设置 `LCAM_ROUND=v3_gemma` 后沿用 v2 的脚本，配置在 `configs/v3_gemma/`。不设置时一切照旧是 v2。
>
> **v2 训练批次（A_base / AG / A_compute，各 3 个种子）**：操作说明见 [docs/v2/README.md](docs/v2/README.md)，用 `scripts/train_v2.py`、`configs/v2/` 和 `scripts/run_v2.sh`。下文描述的是 v1（B0/B1/B2，2026-10-19/20 已跑完），v1 代码保持原样，不再修改。

这是 Locus 相机平移基准（协议 `locus-cam-v1.7.0`）的训练代码。模型是 **Qwen3.5-35B-A3B**：一个多模态 MoE，总参数 35B，每个 token 激活 3B，由 30 层 Gated DeltaNet 和 10 层全注意力交替组成，每层后面接 256 个专家（每次用 8 个，另加 1 个共享专家）。这个选择定于 2026-10-16，替换原先的 Qwen2.5-VL-7B。实验分三组：

- **B0**：不训练，直接用原始模型；
- **B1**：只训练"看图给出动作"（Action-SFT）；
- **B2**：在 B1 的基础上加一项"看图给出几何"（Geometry-SFT），两类样本按 2:1 混合。

核心比较是 B2 对 B1。测试时三组条件相同：只给主画面和俯视图，提交一次动作。

训练和推理都**关闭推理模式**（`enable_thinking=False`），模型直接输出 JSON。B0 另外在开启推理模式下测一遍，作为参照，但不拿来和 B1/B2 比。

这个仓库只放代码和配置。数据由基准目录里的 `training_exports/make_server_bundle.py` 打成一个包，单独传到服务器。

**状态：完整流程已在本机用缩小版模型跑通，但还没有在 GPU 上用真实权重跑过。** 缩小版模型用的是真实的模型类、分词器、图像处理器和对话模板，层的排布和 MoE 结构也和原模型一致，只是层数和维度缩小、权重随机。依次跑通了下面这些：

- **数据核对**：真实样本截取的小包，24 条全部编码成功；被 mask 的几何字段确实没有计入 loss。
- **前向检查**：loss 为 12.43，正好等于 ln(词表大小)，说明标签和 mask 对齐无误。
- **显存探针**：LoRA 正确加在两种注意力层和共享专家上，路由专家、router、视觉部分都冻结；反向传播正常。
- **训练**：B1 和 B2 各跑若干步。计划曝光次数与实际一致（B2 是 2:1），检查点按动作曝光次数保存，LoRA 权重确实被更新。
- **过拟合测试**（`tests/mock/overfit_check.py`）：训练后，用推理脚本对同样的提示词生成，能一字不差地输出目标答案，并在 `<|im_end|>` 处停止。这说明训练时拼的提示词和推理时完全一致。
- **推理和选检查点**：带适配器和不带适配器、有图和无图、开启推理模式都能生成；输出文件能直接交给基准评分器打分。

**没有测到的部分**：`scripts/infer_vllm.py`（vLLM 不支持 macOS）；`flash-linear-attention` / `causal-conv1d` 两个加速内核；真实权重下的显存和速度；多卡切分（`device_map: auto`）。

**上服务器前，先用缩小版模型做一遍冒烟测试**（几分钟就能跑完，不需要下载权重）：

```bash
python tests/mock/make_tiny_model.py <只含 Qwen3.5-35B-A3B 配置和分词器文件的目录> /tmp/tiny-model
```

```bash
python tests/mock/make_mini_bundle.py /data/locus-bundle /tmp/mini-bundle
```

然后把一份 `configs/common.yaml` 里的 `model.id` 指向 `/tmp/tiny-model`，按下面的第 1–6 步对 `/tmp/mini-bundle` 跑一遍，再运行：

```bash
python tests/mock/overfit_check.py --config <这份配置所在的目录> --work /tmp/overfit
```

## 目录

| 路径 | 作用 |
|---|---|
| `configs/` | `common.yaml` 放共用设置；`B0/B1/B2.yaml` 放每组的数据配比和预算，含两种公平口径 |
| `lcam/spans.py` | 找出几何标签里每个字段在文本中的位置，被 mask 的字段不计 loss |
| `lcam/data.py` | 读取导出文件（只读明确列出的文件），检查跨分区隔离，按种子生成采样计划并统计曝光次数 |
| `lcam/encode.py` | 编码对话模板、图片和标签：只训练回答部分；拼接对不上或超长时直接报错，不截断 |
| `lcam/model.py`、`lcam/targets.py` | 加载模型。LoRA 只加在两种注意力层和共享专家上，路由专家、router、视觉塔和词表都冻结。层名是从加载后的模型里按名称识别的，不写死；选中了哪些层会写进 run.json 和 forward_check 的输出，**训练前必须人工核对一遍** |
| `lcam/parse.py` | 与基准评分器相同的答案解析规则（P1–P4） |
| `scripts/run_pipeline.sh` | 总控脚本：在服务器上按阶段串起下面所有脚本，见下一节 |
| `scripts/check_data.py` | 用真实 processor 核对每条样本的长度、视觉 token 数和 mask 情况（不需要权重和 GPU） |
| `scripts/forward_check.py` | 只做前向计算并输出 loss，不更新权重；backward 显存探针需要单独授权 |
| `scripts/train.py` | 正式训练，必须加 `--allow-optimizer-step` 才会更新权重 |
| `scripts/infer.py` | 用 transformers 贪心解码生成答案，输出格式与基准评分器的提交格式一致 |
| `scripts/infer_vllm.py` | 用 vLLM 推理，速度快，B0 开启推理模式时用它；关闭推理时用贪心解码，开启时按模型卡推荐的采样参数 |
| `scripts/select_checkpoint.py` | 只根据验证集选检查点 |
| `tests/mock/` | 缩小版模型、迷你数据包、冒烟测试配置、过拟合一致性测试，用于不下载权重的冒烟测试 |

## 服务器上一条龙：`scripts/run_pipeline.sh`

先做三件准备：
1. 把数据包 `locus-bundle-v1.7.0.tar` 传到服务器，核对 sha256 后解压到 `/data`；
2. 克隆本仓库；
3. 设置两个环境变量：

```bash
export BUNDLE=/data/locus-bundle WORK=/data/locus-work
```

然后按阶段运行。每个阶段的日志写在 `$WORK/logs/` 下，中断后可以重跑，已完成的推理结果和训练会保留。

| 阶段 | 做什么 | 是否需要授权 |
|---|---|---|
| `setup` | 装依赖，下载 Qwen3.5-35B-A3B，并把实际下载到的版本号写进 `configs/common.yaml` | 否 |
| `smoke` | 用缩小版随机模型加迷你数据包，把每个脚本都跑一遍，包括过拟合一致性测试。几分钟跑完 | 否 |
| `check` | 用真实 processor 核对训练集和验证集的长度、视觉 token 和 mask | 否 |
| `forward` | 用真实模型只做前向、算 loss，不更新权重 | 否 |
| `probe` | 做一次反向，测峰值显存 | `ALLOW_BACKWARD_PROBE=1` |
| `b0` | 未训练模型跑验证集和全部测试提示词（关闭推理），另跑一遍开启推理的 `test_V0` | 否 |
| `train` | 训练 B1/B2：按 `$BUDGETS` 两种口径 × `$SEEDS` 各个种子 | `ALLOW_TRAIN=1` |
| `select` | 每个检查点都跑验证集，只按验证集成功率选出最好的一个 | 否 |
| `test` | 每个训练组用选出的检查点，把全部测试提示词跑一次 | 否 |
| `pack` | 把答案、训练记录、选择结果和选中的适配器打成一个 tar，发回基准方打分 | 否 |
| `all` | 依次跑 `setup smoke check forward b0`；只有设置了 `ALLOW_TRAIN=1`，才继续跑 `train select test pack` | — |

例如先跑到训练之前，把结果发回来确认：

```bash
scripts/run_pipeline.sh all
```

确认没有问题、并得到训练授权后：

```bash
ALLOW_TRAIN=1 SEEDS="0 1 2" BUDGETS="primary_fixed_total secondary_matched" scripts/run_pipeline.sh train
```

```bash
scripts/run_pipeline.sh select && scripts/run_pipeline.sh test && scripts/run_pipeline.sh pack
```

可选环境变量：
- `ARMS`：要训练哪几组，默认 `"B1 B2"`；
- `MICRO_BATCH`：每次前向的样本数，默认 4；
- `WORKERS`：数据加载进程数，默认 4；
- `ENGINE_B0`：B0 用什么推理，默认 vLLM；
- `ENGINE_ADAPTER`：带适配器时用什么推理，默认 transformers；
- `TP`：vLLM 张量并行的卡数；
- `PY`：Python 解释器路径。

测试集答案只在基准方的机器上打分，服务器上没有测试集的答案表。

## 各脚本单独使用（`run_pipeline.sh` 就是按这个顺序调用它们）

每完成一步，把输出的 json 发回来确认，再做下一步。

```bash
pip install -r requirements.txt
```

```bash
huggingface-cli download Qwen/Qwen3.5-35B-A3B
```

下载完成后，把实际下载到的 commit hash 填进 `configs/common.yaml` 的 `model.revision`。

```bash
tar -xf locus-bundle.tar -C /data
```

```bash
python3 -m unittest discover -s tests
```

**第 1 步：数据核对。** 用真实 processor 统计长度、视觉 token 和 mask，不加载模型权重：

```bash
python scripts/check_data.py --config configs/common.yaml --bundle /data/locus-bundle
```

**第 2 步：前向检查。** 只做 forward，不创建优化器：

```bash
python scripts/forward_check.py --config configs/common.yaml --bundle /data/locus-bundle --batches 4 --micro-batch 4
```

**第 3 步（需要单独授权）：显存探针。** 做一次 backward 来测峰值显存，梯度用完即丢弃：

```bash
LCAM_ALLOW_BACKWARD_PROBE=1 python scripts/forward_check.py --config configs/common.yaml --bundle /data/locus-bundle --batches 2 --micro-batch 4 --backward-probe
```

**第 4 步：B0 基线。** 不训练，直接用原始模型生成答案。先测关闭推理模式的版本，这是和 B1/B2 比较的那个：

```bash
python scripts/infer_vllm.py --config configs/common.yaml --bundle /data/locus-bundle --prompts eval_prompts/test_V0.jsonl --out outputs/B0-nothink/test_V0.jsonl
```

再测开启推理模式的版本，作为参照：

```bash
python scripts/infer_vllm.py --config configs/common.yaml --bundle /data/locus-bundle --prompts eval_prompts/test_V0.jsonl --out outputs/B0-think/test_V0.jsonl --thinking
```

其他条件换成 `eval_prompts/` 下对应的文件。生成的答案文件拿回基准机器打分，服务器上没有测试集的答案表。

**第 5 步（需要授权后才能开训）：B1 / B2 训练。** 先不加 `--allow-optimizer-step` 跑一遍，确认采样计划正确：

```bash
python scripts/train.py --config configs/B1.yaml --budget primary_fixed_total --seed 0 --bundle /data/locus-bundle --out runs/B1-primary-s0
```

确认无误后，加上该参数正式训练：

```bash
python scripts/train.py --config configs/B1.yaml --budget primary_fixed_total --seed 0 --bundle /data/locus-bundle --out runs/B1-primary-s0 --allow-optimizer-step
```

**第 6 步：用验证集选检查点。** 每个检查点都要先跑验证集：

```bash
python scripts/infer.py --config configs/common.yaml --bundle /data/locus-bundle --prompts eval_prompts/validation_V0.jsonl --out runs/B1-primary-s0/val/ckpt-1.jsonl --adapter runs/B1-primary-s0/checkpoints/ckpt-1
```

全部检查点跑完后，选出最好的一个：

```bash
python scripts/select_checkpoint.py --bundle /data/locus-bundle --run runs/B1-primary-s0
```

选定之后，只用这个检查点在测试集上跑一次。

## 规则（与基准任务书一致）

- **检查点选择**：只看验证集上的成功率，测试集不参与选择。
- **数据来源**：B1/B2 只用 `action_sft` 和 `geometry_sft`；`outcome_records` 和 `geometry_sft_privileged` 由代码拒绝读取。
- **种子数**：在第一次正式训练之前定下来，不看结果再追加。
- **曝光次数**：每个检查点都记录各类数据的实际曝光次数，写在 `exposures.json` 和 `metrics.jsonl` 里。
- **解码**：贪心解码，同一提示词重复采样的结果完全相同，所以只算一次，不当作多个独立样本。

## 硬件

- **显存**：bf16 权重约 70 GB。LoRA 训练建议一张 H200（141 GB），或两张 80 GB 卡（`device_map: auto` 会把各层分到两张卡上）。峰值显存以第 3 步实测为准。
- **速度**：Gated DeltaNet 层如果没装 `flash-linear-attention` / `causal-conv1d` 内核，会退回很慢的纯 torch 实现。训练速度同样以实测为准，再据此确定种子数和预算。
