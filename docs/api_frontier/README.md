# 国际一线模型零样本评测（只推理，不训练）

用于论文主表补充国际一线模型。不需要 GPU，任何能访问这几家 API 的机器都能跑，需要代理时设置 `HTTPS_PROXY`。

## 测哪些模型

默认每家取比最新旗舰低一档的模型。型号和价格已于 2026-10-30 对照各家官方模型页和价目表核对。

| 厂商 | 默认测 | 每百万 token 价格（输入 / 输出） | 说明 |
|---|---|---|---|
| OpenAI | `gpt-5.6-sol` | 4 / 20 美元（优惠价，至少持续到 2026-11-21） | 上一代旗舰，比最新的 GPT-6 低一档；默认推理强度 medium |
| Google | `gemini-3.1-pro-preview` | 2 / 12 美元（提示 20 万 token 以内） | 上一代 Pro，比 Gemini 4 Argon 低一档；目前仍是预览版，Google 可能调整或下线 |
| Anthropic | `claude-opus-5-5` | 4 / 20 美元 | 比 Claude Fable 5.1 低一档，也是 Anthropic 的默认推荐型号；默认推理强度 medium |

脚本里同时保留了各家最新旗舰（`gpt-6-astra`、`gemini-4-argon`、`claude-fable-5-1`）和更便宜的型号，用 `--models` 指定即可。

## 测哪些题

- 默认只跑主测试 600 题（`test_V0`）。论文主线"零样本模型在已见结构族和留出结构族上成绩持平"只需要这一组。
- 加上 `test_diagnostic_S`（把精确几何写进题目，120 题），可以给诊断章节补一个数据点。

提示词就是数据包里的 `eval_prompts/*.jsonl`。题目文字、系统提示和图片顺序，已逐条核对与主表 4 个 API 模型收到的完全一致（`test_V0` 600 条、`test_diagnostic_S` 120 条）。

## 设置

- **请求方式**：每题单独请求，不加示例、不加工具、不预填，采样参数用各家默认值。
- **输出上限**：取各家接口允许的大值（OpenAI、Anthropic 128000，Google 65536），避免答案被上限截断。
- **推理强度**：默认用各家的默认值，写进每条答案的 `effort` 字段。要统一设置时加 `--effort`。注意 Claude Opus 5.5 的默认强度是 medium。
- **Anthropic 不开拒答回退。** 开了回退的话，被拒的请求会自动换另一个模型重答，答案就不是被测模型的了。被拒的题按原样记录（`finish` 为 `refusal`），算作未答对。

## 运行

```bash
pip install openai anthropic google-genai
```

先看请求数和费用估算，这一步不发任何请求：

```bash
python scripts/run_api_models.py --bundle $BUNDLE --plan --prompts test_V0,test_diagnostic_S
```

正式运行，key 只从环境变量读取：

```bash
OPENAI_API_KEY=... GEMINI_API_KEY=... ANTHROPIC_API_KEY=... python scripts/run_api_models.py --bundle $BUNDLE --out $LOCUS_ROOT/work/api --prompts test_V0
```

- **输出**：`work/api/<模型>/<提示文件>.jsonl`，格式与 `scripts/infer.py` 相同，另外记录了用量和耗时。
- **续跑**：重跑同一条命令会跳过已完成的题。
- **错误处理**：只把模型的真实回答写进结果文件。
  - 余额不足、账单问题，或限流在 2–16 分钟的递增等待后仍未恢复，会停下（退出码 3），不写答案。
  - 其他错误重试 4 轮后仍失败的，记入 `errors.jsonl`，下次重跑时自动补跑（仍有剩余时退出码为 2）。

跑完后把 `work/api/` 整个目录发回基准方打分。

## 费用估算

按每题输入 2000、输出 12000 个 token 估算。主表 4 个推理模型实测每题输出 6.5k–17k 个 token。

| 模型 | V0 600 题 | 加 S 条件共 720 题 | 输出量为一半到两倍时（720 题） |
|---|---:|---:|---:|
| gpt-5.6-sol | 约 149 美元 | 约 179 美元 | 92–351 美元 |
| gemini-3.1-pro-preview | 约 89 美元 | 约 107 美元 | 55–210 美元 |
| claude-opus-5-5 | 约 149 美元 | 约 179 美元 | 92–351 美元 |
| **合计** | **约 386 美元** | **约 464 美元** | |
