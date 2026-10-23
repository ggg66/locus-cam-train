# v2 训练批次：服务器操作说明

v2 比较三组训练，每组 3 个种子，共 9 次训练：

| 组 | 训练什么 | 曝光次数 |
|---|---|---|
| A_base | 只学动作 | 动作 15000 |
| AG | 动作加几何 | 动作 15000（同种子与 A_base 顺序相同）+ 几何 5000 |
| A_compute | 只学动作，训练计算量与 AG 相同 | 动作次数由标定得出，前 15000 次与 A_base 相同 |

B0（未训练底座）只做推理。协议见 [protocol_v2.yaml](protocol_v2.yaml)，审计记录见 [training_audit.md](training_audit.md)。

v1 的代码 `scripts/train.py` 和 `configs/B*.yaml` 保持原样；v2 用 `scripts/train_v2.py` 和 `configs/v2/`。

## 一、准备

1. **拉取本仓库。** 放在共享盘上，每个节点都能用同一路径访问到。
2. **在原 v1 数据包上解压增量包** `locus-bundle-v1.7.0-addon-v2.tar`（245 MB，由基准方单独传过去）。增量包只新增文件，不改动 v1 数据包里的任何内容。
3. **设置环境变量。** 路径不写进仓库，`LOCUS_ROOT` 下的目录结构沿用 v1：`venv/`、`data/locus-bundle/`、`models/Qwen3.5-35B-A3B/`、`work/`。

```bash
export LOCUS_ROOT=/共享盘/项目目录
```

```bash
export ADDON_TAR=$LOCUS_ROOT/data/locus-bundle-v1.7.0-addon-v2.tar ADDON_SHA256=abaf2fe1edd90abece98154845078a13be424c6a95702f7333f4c0fc68ec3cc8
```

如果权重或数据包不在默认位置，再设置 `LCAM_MODEL`、`BUNDLE`、`PY`，见 `launch/env.sh`。

## 二、训练前的门禁（按顺序跑，结果都在 `$LOCUS_ROOT/work/v2/audit/`）

```bash
scripts/run_v2.sh addon unit smoke check stats audit
```

- `addon`：核对增量包的 sha256 后解压，再逐个核对文件。
- `unit`：单元测试，包括损失公式在不同切分下梯度一致。
- `smoke`：用缩小版随机模型把每个 v2 脚本走一遍，包括单卡和 2 卡。用到两张 GPU，几分钟跑完。
- `check`：用真实 processor 核对 v2 的训练集和验证集。
- `stats`：统计每条样本的 token 数。用 16 个 CPU 进程，不需要 GPU。
- `audit`：
  - 核算 v1 的损失权重（不用 GPU）；
  - 用真实模型核对 v1 的损失里有没有额外项（单卡）；
  - 输出 A_base 和 AG 的批次追踪。

```bash
scripts/run_v2.sh equiv short calibrate
```

- `equiv`：AG 种子 0 的第一步，分别用单卡（累积 32 次）和 8 卡（累积 4 次）算梯度，关掉 dropout，两者必须一致。
- `short`：A_base 和 AG 各在 8 卡上跑 12 步，测每步用时；AG 保存一次适配器，再重新加载核对。
- `calibrate`：按含补齐的 token 数，算出 A_compute 的动作次数，并列出按真实 token 数和按 GPU 时间算会差多少。这一步只打印结果，不冻结。

把 `work/v2/audit/` 整个目录发回基准方。逐项对照 [training_audit.md](training_audit.md) 第三部分的通过标准，全部通过后再冻结标定：

```bash
scripts/run_v2.sh freeze
```

冻结以后，把 `configs/v2/calibration.json` 提交进仓库，三个种子都用这一个数。

## 三、正式训练（需要授权）

多节点时，每个节点跑一组，节点 IP 每行一个：

```bash
ALLOW_TRAIN=1 NODES=$LOCUS_ROOT/work/v2/nodes.txt scripts/run_v2.sh train
```

单节点时去掉 `NODES`，9 次训练依次跑。查看进度：

```bash
launch/status.sh
```

按 v1 的速度，每次训练在 8 张 H20 上约 25–40 分钟。每次训练都会在进度 25%、50%、75%、100% 处保存检查点，100% 的那个是预定终点。

## 四、评测与打包

```bash
NODES=$LOCUS_ROOT/work/v2/nodes.txt scripts/run_v2.sh val test pack
```

- `val`：每个检查点跑验证集，按验证集成功率选出最好的一个。这个结果单独报告，不替代预定终点。
- `test`：每次训练的预定终点和 B0，都跑同一套评测：
  - 600 道主测试题；
  - 240 个变体；
  - 275 道同图换目标；
  - 120 道几何预测；
  - 120 道 12 候选后果面板。
  B0 另跑 dev 上的示例照抄对照。如果验证最优不是终点，再用验证最优补跑一遍。
- `pack`：把答案、训练记录、各检查点的曝光和探针损失、审计结果打成一个 tar，不含适配器。发回基准方打分。

服务器上没有测试集的答案表，全部打分都在基准方的机器上完成。

## 五、本批不做的事

后果监督（B3）、RL/DPO、加大 LoRA 或解冻视觉塔、扫描混合比例、生成新的训练场景、改动测试集，都不在本批范围内。9 次训练和评测跑完，这一批就结束，不会因为哪一组赢了再追加调参。
