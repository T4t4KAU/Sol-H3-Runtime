# SOL-H3 profiling 交接

更新日期：2026-11-03。本文从已获得 GPU、进入 GPU 服务器／容器终端开始，不包含服务器认证或资源申请步骤。本轮完成文件整理和 profiling 工具准备，没有新申请 GPU，也没有新增整模型测量结果。

## 1. 当前状态与保留目录

2026-11-03 核查时，原八卡 API 分配 `41dab198-08d6-445c-8cd5-b7c42b6d0f4c` 已结束，当前账号没有运行中的 GPU 会话。API 曾通过真实请求验收，但当前不在运行。历史 `api.pid` 等文件已标为 terminated，不能当成当前进程 PID 使用。

| 用途 | 服务器路径 |
|---|---|
| 整理后的代码 | `/labdata2/hwx/Sol-H3-Runtime` |
| 公共模型 | `/Datasets/public_models/SOL-H3/MiniMax-H3` |
| 公共 FastH3 LoRA | `/Datasets/public_models/SOL-H3/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors` |
| 公共解释器 | `/Datasets/public_models/SOL-H3/envs/solh3-4090/bin/python` |
| 公共基础 Python | `/Datasets/public_models/SOL-H3/python/cpython-3.13.9-linux-x86_64-gnu` |
| 历史实验与 trace | `/labdata2/hwx/sol-h3-8xrtx4090/` |
| API 与功能验收证据 | `/labdata2/hwx/sol-h3-8xrtx4090/api-service-20261102T034900Z/` |
| 本轮清理凭据 | `/labdata2/hwx/sol-h3-8xrtx4090/cleanup-receipts/20261103-handoff/` |
| 建议的新 profiling 输出根目录 | `/labdata2/hwx/sol-h3-8xrtx4090/profiles/` |

GPU 环境需要同时挂载代码／结果所在的 `/labdata2/hwx` 和公共资产所在的 `/Datasets/public_models`。旧 checkpoint 路径改为公共副本的软链接后，也要求公共目录在容器内可见。公共权重和环境不要原地升级。

## 2. 先确认要测哪条路径

| 路径 | 已核实的配置与范围 | 使用建议 |
|---|---|---|
| 本仓库八卡显存适配入口 | FastH3 四次 DiT 前向；1344×768、124 帧、24 FPS；8×4090 24 GiB；FSDP8＋SP8；dense Flash SDPA；分阶段加载；eager tiled VAE | 可以沿用已完成视频生成验证的配置，按第 4 节采集新 trace |
| 已有 native 全优化研究 trace | 2／4×4090；FastH3 四次前向；Sol-Attn Triton；原生融合与 AdaLN；SP2／SP4；无 FSDP；20 个常驻 block＋30 个固定 CPU offload block；真实文本条件缓存；FP8 输出通信 | 先读第 7 节已有证据；这是 DiT-only 研究路径，不是上面八卡完整生成配置 |
| 原样官方实现 | `official/Sol-H3/`，固定上游 revision `781e8341fb3b29d3aea5d50265772e9619340b66` | 不因设置多个 SP rank 就认为权重被分片；在 24 GiB 卡上不可直接替代显存适配入口 |

八卡显存适配默认没有开启稀疏注意力、AdaLN 预计算、packed SP 和原生 kernel fusions，其 trace 不能代表官方全优化计算。不要为了“能跑”临时改精度、分辨率或算子后端，然后仍用原来的配置名称汇报。

四步、原版 49 步、H3 四步＋LTX 三步是不同工作负载。不要混用数据；50 个 transformer block 也不等于 50 次去噪。

## 3. 入口能力与本次验证边界

当前 `adapters/rtx4090/infer.py` 的内置 `--profile-dir`／`--benchmark-dir` 只接受 `original` 或 `two-stage` 模式，不能直接用于 `single-stage`。官方 `infer.py` 也没有这两个参数。给现有四步启动脚本追加 `--profile-dir` 会在参数检查阶段失败。

本轮新增独立包装器 `tools/profile_single_stage.py`：通过现有 instrumentation 接口观测四步路径，不改动官方与冻结适配源码。它可以：

- 选择 DiT 前向索引和采集 rank，导出 CPU/CUDA Chrome trace 与算子统计。
- 在无 profiler 的独立 benchmark 运行中，同步计时 staged `denoise` 阶段。
- 记录四次前向的完成数、源码哈希、运行参数、每 rank 状态与错误。

**包装器已通过 CPU 参数及范围检查，尚未在真实 GPU 上验收这次新增的 CUDA 采集／计时接入。** 已有 GPU 功能验证属于底层四步推理脚本；已有 native trace 属于第 7 节旧研究路径。下一位拿到 GPU 后，应先执行下面的 CUPTI 小测试，再验证新包装器的前向计数、trace 与视频输出，不能把本交接当成新增 profiling 成功证据。

## 4. 当前八卡四步路径的采集步骤

### 4.1 固定路径、参数和证据目录

```bash
export SOL_H3_REPO=/labdata2/hwx/Sol-H3-Runtime
export SOL_H3_ASSETS=/Datasets/public_models/SOL-H3
export SOL_H3_PYTHON="$SOL_H3_ASSETS/envs/solh3-4090/bin/python"
export SOL_H3_MODEL="$SOL_H3_ASSETS/MiniMax-H3"
export SOL_H3_ADAPTER="$SOL_H3_ASSETS/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"
export RUN_ROOT="/labdata2/hwx/sol-h3-8xrtx4090/profiles/fast4-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
cd "$SOL_H3_REPO"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv > "$RUN_ROOT/gpus.csv"
nvidia-smi topo -m > "$RUN_ROOT/topology.txt"
"$SOL_H3_PYTHON" tools/verify_sources.py > "$RUN_ROOT/source-integrity.json"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DISABLE_ADDMM_CUDA_LT=1

H3_ARGS=(
  --model "$SOL_H3_MODEL" --adapter "$SOL_H3_ADAPTER"
  --task t2v --duration 5 --seed 0
  --prompt 'An adult dancer performs energetic choreography on a brightly lit stage.'
  --mode single-stage
  --fsdp-degree 8 --ulysses-degree 8 --staged-loading --text-fsdp --flash-sdpa
  --no-tf32 --no-bf16-reduced-precision-reduction --no-adaln-precompute
  --no-kernel-fusions --no-packed-sp --no-parallel-vae
  --vae-compile-mode eager --vae-tile-size 256
  --attention-backend dense --compute-quant none --lora-mode merged
)
```

这组参数与已验证的 `adapters/rtx4090/run_single_stage_8x4090.sh` 一致。这里只手动展开启动参数，便于在每个 worker 外增加观察器，不要再套一层八卡启动脚本。不要覆盖调度器已经设置好的 GPU 可见性；先确认当前环境确实提供八张目标 GPU。

### 4.2 检查解析和 CUPTI 可用性

```bash
"$SOL_H3_PYTHON" tools/profile_single_stage.py \
  --artifacts "$RUN_ROOT/argument-check" --check-arguments \
  -- "${H3_ARGS[@]}" --output "$RUN_ROOT/argument-check.mp4"

"$SOL_H3_PYTHON" - "$RUN_ROOT/cupti-smoke.json" <<'PY'
import json
from pathlib import Path
import sys
import torch

assert torch.cuda.is_available(), "No CUDA device"
x = torch.randn(512, 512, device="cuda")
x @ x
torch.cuda.synchronize()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False, profile_memory=False, with_stack=False,
) as prof:
    x @ x
    torch.cuda.synchronize()
prof.export_chrome_trace(sys.argv[1])
events = json.loads(Path(sys.argv[1]).read_text())["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel"]
assert kernels, "No CUDA kernel events: check CUPTI and profiling permissions"
print("PASS: CUDA kernel events", len(kernels))
PY
```

只有 CUDA kernel 事件存在才继续；CPU-only trace 不能用于 GPU 算子结论。不要在公共环境原地执行 `pip install -U` 解决问题，先检查当前 CUDA、CUPTI、驱动和容器权限。

### 4.3 有界算子 profiling

首次先只采 rank 0，覆盖四次 DiT 前向。其余 rank 仍参与原计算与通信，只记录前向计数和状态：

```bash
set -o pipefail
"$SOL_H3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/profile_single_stage.py \
  --artifacts "$RUN_ROOT/profile" --capture profile \
  --forward-indices 0,1,2,3 --profile-ranks 0 \
  -- "${H3_ARGS[@]}" --output "$RUN_ROOT/profile.mp4" \
  2>&1 | tee "$RUN_ROOT/profile.log"
```

每个 `rankN/` 目录包括：

| 文件 | 含义 |
|---|---|
| `metadata.json` | 源码哈希、启动参数、rank、torch/CUDA 身份与采样范围 |
| `summary-rankN.json` | 前向计数、状态、阶段 host wall time、allocator 峰值 |
| `launch-result.json` | 该 worker 完整推理入口是否退出成功，包含编码之后的状态 |
| `trace-rank0-forward0.json` 等 | 被采 rank 的选定前向 CPU/CUDA trace |
| `operators-rank0-forward0.json` 等 | PyTorch operator 与 GPU 事件统计，单位见 JSON 字段 |

trace 从 root forward 的 pre-hook 开始，覆盖该 forward 的 FSDP unshard 和 SP 通信，不仅是矩阵乘。初始化 checkpoint 加载、文本编码、VAE 解码和 MP4 编码不在这些 forward trace 内。

默认不启用 `record_shapes`、memory profiling 或 stack，避免一次收集过多数据。需要确认跨 rank 等待时，另开一个小范围诊断运行，例如 `--forward-indices 2 --profile-ranks 0,1,2,3,4,5,6,7`；这会增加开销，不能作为干净性能结果。不要将不同运行的 trace 拼接为同一个请求。

### 4.4 独立的 DiT 计时

没有可用的同配置干净计时记录时，再单独运行一次；若已有完整记录则复用，不为凑样本重复成功配置。

```bash
set -o pipefail
"$SOL_H3_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/profile_single_stage.py \
  --artifacts "$RUN_ROOT/benchmark" --capture benchmark \
  -- "${H3_ARGS[@]}" --output "$RUN_ROOT/benchmark.mp4" \
  2>&1 | tee "$RUN_ROOT/benchmark.log"
```

这一模式不创建 `torch.profiler`，不做 latent fingerprint 拷贝，仍保留计数 hook、边界同步和状态记录。`timings.synchronized_denoise_s` 的边界是：staged 初始权重加载完成后，进入 `denoise` body 前进行 barrier／CUDA synchronize；body 返回后 CUDA synchronize，再停止计时。它包括四次 DiT 前向及其中的 FSDP／SP 工作，排除初始 checkpoint／LoRA 加载、文本编码、模型释放、VAE 和 MP4 编码。

`timings.synchronized_generate_s` 覆盖完整 `engine.generate()`，包括请求内组件加载、文本编码与解码，排除 engine 构造和 MP4 编码；它不是 DiT 时延。每 rank 的同步 DiT 时间分别保存，报告所有 rank，并以最慢 rank 作为这次同步阶段观测值，不累加或平均各 rank 来冒充请求时延。

### 4.5 验收新采集

```bash
"$SOL_H3_PYTHON" - "$RUN_ROOT/profile" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for rank in range(8):
    folder = root / f"rank{rank}"
    summary = json.loads((folder / f"summary-rank{rank}.json").read_text())
    launch = json.loads((folder / "launch-result.json").read_text())
    assert summary["status"] == launch["status"] == "completed", rank
    assert summary["expected_forwards"] == summary["completed_forwards"] == summary["attempted_forwards"] == 4, rank
for index in range(4):
    path = root / "rank0" / f"operators-rank0-forward{index}.json"
    item = json.loads(path.read_text())
    assert item["forward_index"] == index
    assert any(e["category"] == "kernel" and e["count"] > 0 for e in item["gpu_events"]), index
print("PASS: all ranks completed four forwards; rank0 has four CUDA windows")
PY
```

再完整解码 `profile.mp4`：检查 1344×768、124 帧、24 FPS，视频与音频均非空，方法见部署指南。只有 trace、每 rank 状态、完整输出三者都通过，才把这次采集标为 completed。失败时保留整个目录，不要删掉不完整运行再只汇报成功项。

## 5. 从 trace 读出 DiT 瓶颈

先打开一份前向 trace，观察 CUDA streams、CPU 发射、FSDP all-gather、SP all-to-all 与计算的关系。Chrome trace 可由 [Perfetto UI](https://ui.perfetto.dev/) 打开；工具说明见 [Perfetto 官方文档](https://perfetto.dev/docs/visualization/perfetto-ui)。

下面从 rank 0 的已采前向中提取 GPU kernel 排名，明确将 NCCL 与 memcpy 分开：

```bash
"$SOL_H3_PYTHON" - "$RUN_ROOT/profile/rank0" <<'PY'
from collections import defaultdict
import json
from pathlib import Path
import sys

kernels = defaultdict(float)
other = defaultdict(float)
files = sorted(Path(sys.argv[1]).glob("operators-rank0-forward*.json"))
assert files, "No operator artifacts"
for path in files:
    for event in json.loads(path.read_text())["gpu_events"]:
        seconds = event["duration_us"] / 1e6
        if event["category"] == "kernel" and "nccl" not in event["name"].lower():
            kernels[event["name"]] += seconds
        else:
            other[event["category"] + ("/NCCL" if "nccl" in event["name"].lower() else "")] += seconds
print("Selected-forward kernel duration SUM, seconds; not wall latency")
for name, seconds in sorted(kernels.items(), key=lambda item: -item[1])[:30]:
    print(f"{seconds:.6f}\t{name}")
print("Other recorded duration sums:", dict(other))
PY
```

优先核查 FFN up/down projection、QKV／输出投影、attention kernel、norm／activation／layout，以及这些计算之间的空隙。用 CPU operator／correlation 对应真实 CUDA kernel，不能只看到 `aten::linear` 就把所有时间归入 GEMM。对于 fused kernel，不能在没有内部证据时把一个 kernel 的时间再拆成多个互斥模块。

本包装器使用已有 forward 观察器，没有 native 研究 trace 中的逐模块 `H3_NATIVE/...` 标注。若需要层／模块定位，先使用第 7 节已带标注的 trace；对当前八卡路径扩展标注时只增加观测范围，另存源码身份，不顺便更换算子或调度策略。

## 6. 计时与比较口径

| 指标 | 可回答的问题 | 不能当成 |
|---|---|---|
| benchmark `synchronized_denoise_s` | 当前四次去噪及其分布式工作的同步阶段时延 | 纯计算 kernel 时间，或包含加载／VAE 的整请求耗时 |
| `synchronized_generate_s` | 完整 generate 范围的同步时间 | DiT-only 时延 |
| trace kernel duration sum | 某组被记录 GPU 事件的累计工作量 | 墙钟时延，或端到端时间占比 |
| 同一 rank trace 的区间并集 | 该 GPU 上被选事件覆盖的时间 | 多 rank 全局关键路径 |
| `phases[*].host_wall_s` | 阶段诊断定位 | 保证互斥、同步的 GPU 阶段分解 |
| allocator 峰值 | 从 recorder 初始化到 finalize 的框架分配峰值，含文本／DiT／VAE | 整机显存总量或纯 DiT 峰值 |
| API job timestamps | 包含加载、生成、编码的任务总时间 | DiT 算子 profiling |

profile 和 benchmark 必须分开。只采 rank 0 时，其他 rank 的 summary 即使显示 `mode=benchmark`，也属于同一个受 profiling 扰动的运行；不能把它们当作独立干净基准。

kernel、NCCL 和 memcpy 可在不同 stream 重叠，不相加成总时长。若剔除 NCCL／memcpy 后计算占比很大，只能说明计算 kernel 累计时间的构成；同一时刻的通信仍可能影响计算速度。

比较前固定 prompt、seed、步数、分辨率／帧数、权重／LoRA、精度、attention backend、融合／量化、FSDP／SP、offload、缓存和 warmup 策略，并记录 GPU UUID／拓扑、torch／CUDA／驱动身份。当前默认单请求；不要把不同批量、49 步或 A800 的结果拿来解释这次 4090 四步路径。

PyTorch 的 profiler 选项、Chrome trace 导出和内存／栈采集开销见 [PyTorch 2.11 profiler 文档](https://docs.pytorch.org/docs/2.11/profiler.html)。另外，CUDA Event 是 GPU 事件计时机制，其 `elapsed_time` 单位为毫秒；它与本包装器边界同步的 host wall time 不同，见 [CUDA Event 文档](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html)。

## 7. 已有 native 全优化 DiT 证据

无需新占 GPU，即可先阅读下列已存在的原始 trace 和元数据：

| 路径（均在历史实验根目录下） | 重点文件与范围 |
|---|---|
| `native-fullopt-4090-20261031T030218Z/` | `rank0.trace.json.gz`、`loaded-rank0.json`、`observer-rank0.json`；2 卡，四次前向，FP8 输出通信 |
| `native-fullopt-g4-4090-20261031T045649Z/` | 同类文件；4 卡，四次前向，FP8 输出通信 |
| `native-compute-4090-20261030T174243Z/` | 同类文件；2 卡，四次前向，BF16 输出通信；约 14 GiB 的目录还含其他中间证据 |

前两个目录的 `loaded-rank0.json` 明确记录 SP2／SP4、`fsdp=false`、20 个常驻 block 与 30 个 offload block；`observer-rank0.json` 记录 `model_forwards=4`、只采 rank 0。文档里的 scope 必须以这些字段为准，不能只看目录名“fullopt”就宣称所有平台上的全部优化都已开启。

`analyze_native_trace_v2.py` 使用 `H3_NATIVE/step...` CPU scopes 和 kernel correlation 做归因，将稀疏路由／统计、稀疏 attention core、FFN、QKV／输出投影等分组。输出 `compute-rank0.json` 中 `fraction_of_compute_kernel_sum` 的分母是计算 kernel 累计时间，排除了 NCCL 和 memcpy，不是请求时延。

需要重新分析时，复制分析脚本到新的目录，链接已有 trace，不覆盖原始结果：

```bash
export OLD=/labdata2/hwx/sol-h3-8xrtx4090/native-fullopt-4090-20261031T030218Z
export ANALYSIS="$RUN_ROOT/existing-native-analysis"
mkdir -p "$ANALYSIS"
cp "$OLD/analyze_native_trace_v2.py" "$ANALYSIS/"
ln -s "$OLD/rank0.trace.json.gz" "$ANALYSIS/rank0.trace.json.gz"
"$SOL_H3_PYTHON" "$ANALYSIS/analyze_native_trace_v2.py" > "$ANALYSIS/analysis.log"
```

已有研究 driver 依赖自己的 `runtime/`、`native-deps/` 和 `conditioning.pt`，不是通用生产入口；不要移动其中单个文件后直接重跑。下一步若研究官方已优化计算，优先基于这些 trace 定位问题，再设计只改变一个因素的验证，避免把显存适配配置与 native 路径的差异归因于某一项优化。

## 8. 常见问题和后续交接

- `--profile-dir` 报 mode 错误：这是旧参数的适用范围限制；四步使用新包装器，不修改冻结源码绕过判断。
- 没有 `kernel` 事件：先做第 4.2 节的 CUPTI 检查。CPU-only trace 不计为 GPU profiling 成功。
- 出现 OOM：保留每 rank 错误和参数。先缩小采集窗口／采样 rank，不擅自改变模型精度、几何或内存策略后继续沿用同一配置名称。
- 收到 `FileExistsError`：输出目录防止混合请求，换新 `RUN_ROOT`，不要删除尚未检查的旧结果。
- 某 rank summary 缺失：整次运行未完成；检查 torchrun 日志和 `launch-result.json`，不要只看 rank 0。
- 在 API 里找不到 profiling 参数：API 的请求 schema 不开放这类控制。本次 profiling 直接从 CLI 启动，避免队列和编码混入 DiT 分析；若 API 正在使用同一组 GPU，先协调停止。
- Nsight：登录节点未发现 `nsys`／`ncu`。新 GPU 环境内应重新核对工具和权限，不能照抄一条 Nsight 命令就假定采集成功；本交接的主路径使用环境中已有的 torch profiler。

交接时至少保留：源码 revision／哈希、模型与公共环境身份、实际运行命令、prompt hash／seed／几何、GPU UUID／拓扑、每 rank 状态、原始 trace、算子 JSON、成功或失败的日志、输出视频与解码检查。没有同一次运行、同一时钟与真实关键路径依据时，不生成端到端百分比图。

## 9. 本轮清理结果

本轮已完成 61 份唯一权重内容的 SHA256 核验，并去重旧路径的重复存储。四个旧目录和 14 个文本编码器分片路径已改为公共副本软链接。按唯一 inode 的原分配块计，回收约 **135.85 GiB** 重复占用，不重复计算硬链接。另清除五项临时上传／失败 clone／更新暂存文件。

历史 trace、实验 JSON／SQLite、API 验收视频与日志、唯一 LTX／upscaler 权重、公共模型及环境均保留。完整证据见清理回执目录中的 `weights-dedupe.json`、`text-encoder-dedupe.json`、`temporary-cleanup.json`、`cleanup-summary.json` 和 `清理记录.md`。
