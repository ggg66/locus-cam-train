# 一线模型的诊断条件（只推理，不训练）

在 V0 跑完之后，补齐几何路线、后果面板、C1、C2，每个模型 600 个请求。提示词、推理强度和输出上限都与 V0 那一轮相同；C2 的提示词由模型自己的 V0 动作生成。

```bash
AIHUB_API_KEY=... python -u scripts/run_api_diagnostics.py --bundle $BUNDLE --v0 $LOCUS_ROOT/work/api --out $LOCUS_ROOT/work/api_diag --gateway aihub
```

重跑同一条命令会跳过已完成的答案。进度看 `work/api_diag/batch_status.json` 和 `work/api_diag/<模型>.log`。跑完把 `work/api_diag.tar.gz` 和同名 `.sha256` 发回打分。
