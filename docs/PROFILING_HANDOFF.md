# SOL-H3 DiT profiling 交接

更新：2026-11-03。目标是分析单请求、FastH3 四次去噪中 DiT 的计算开销。测试方法是**先准备并暂存其他模块提供的输入，复用这些输入，只执行和观测 DiT；去噪结束立即停止，不执行 VAE 解码、音频解码和视频编码**。

本文依据服务器上的实际认证、调度配置和已完成的 native SOL-H3 双卡／四卡 profiling 驱动整理。本次更新核实了认证、路径及驱动实现，没有重新申请 GPU 或新增性能数据。

## 1. 登录服务器与认证调度器

### 1.1 本机通过 SSH 密钥登录

在保存有密钥的本机执行：

```bash
ssh -i /home/hwx/.ssh/id_ed25519_lins_lab \
  -o IdentitiesOnly=yes -p 22332 tempuser16@10.0.2.166
```

登录账号是 `tempuser16`，端口为 `22332`，私钥位置为 `/home/hwx/.ssh/id_ed25519_lins_lab`。接手者使用自己有权使用的 SSH 密钥；不要复制私钥内容到代码仓库或交接文档。

### 1.2 在登录节点认证 Determined

SSH 登录和 GPU 调度器登录是两套认证。调度器使用登录节点上的令牌缓存：`/home/tempuser16/.config/determined/auth.json`。

以下命令均在登录节点执行，显式指定当前配置使用的 master：

```bash
export DET_MASTER=http://192.168.123.168:8080
detcli() { /usr/bin/python3 -m determined.cli -m "$DET_MASTER" "$@"; }
detcli user whoami
```

2026-11-03 实测返回当前账号 `tempuser16`。如果令牌过期，重新登录：

```bash
detcli user login tempuser16
detcli user whoami
```

按交互提示输入账号密码，成功后 CLI 更新令牌缓存。SSH 密钥不能替代此步骤；文档不保存密码或令牌值。不要用其他账号已有的 GPU 会话运行测试。

## 2. GPU 申请、排队与进入容器

### 2.1 选择资源

当前消费卡资源池包括 `64c128t_512_4090`（每节点约 512 GiB 主机内存）和 `128c256t_768_4090`（每节点约 768 GiB 主机内存），均有 RTX4090 24 GiB 节点。资源池名称与实际可用资源仍以调度器当时的配置为准。

已有 DiT profiling 驱动分别支持双卡和四卡，必须使用匹配的驱动。它们使用 SP2／SP4，不分片模型权重；每个 rank 都需要 CPU 权重副本。现有主机内存准入门槛为：双卡 `MemAvailable > 260,000,000,000` 字节，四卡 `> 460,000,000,000` 字节。这是现有驱动的保守门槛，不是模型固有最小内存需求。四卡优先选主机内存更充足的资源池。

### 2.2 提交一次 GPU 会话申请

在登录节点执行。默认申请双卡，改为四卡时同时修改 `SOL_H3_GPUS=4` 并选择对应池：

```bash
export SOL_H3_GPUS=2
export SOL_H3_POOL=64c128t_512_4090
export SOL_H3_REQUEST="$HOME/sol-h3-dit-request.json"
/usr/bin/python3 - <<'PYCONF'
import json, os
from pathlib import Path
n = int(os.environ["SOL_H3_GPUS"])
assert n in (2, 4)
image = "harbor.lins.lab/determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.31.1"
config = {
    "description": f"SOL-H3 DiT profiling {n} GPU",
    "resources": {"resource_pool": os.environ["SOL_H3_POOL"],
                  "slots": n, "shm_size": 68719476736},
    "bind_mounts": [
        {"host_path": "/labdata2/hwx", "container_path": "/labdata2/hwx", "read_only": False},
        {"host_path": "/Datasets/public_models", "container_path": "/Datasets/public_models", "read_only": True},
    ],
    "work_dir": "/labdata2/hwx/Sol-H3-Runtime",
    "environment": {"image": {"cpu": image, "cuda": image}},
}
Path(os.environ["SOL_H3_REQUEST"]).write_text(json.dumps(config, indent=2) + "\n")
PYCONF
detcli shell start --config-file "$SOL_H3_REQUEST" --detach
```

该命令使用现有集群镜像，实际模型 Python 使用第 3 节已有环境。提交后保存输出的 shell/task ID，例如设置 `SOL_H3_SHELL_ID` 为实际返回值。申请使用当前账号，两个挂载分别提供代码／实验数据和公共权重／环境。

### 2.3 排队不等于运行

```bash
detcli shell list
```

拿到任务 ID **不代表已拿到 GPU**。`QUEUED` 表示等待调度；GPU 数量、同节点资源和其他作业均可能影响等待时间，历史上有等待数小时的情况。不要为同一测试重复提交多个申请。资源池总卡数也不等于空闲卡数。

等待目标会话显示 `RUNNING` 后，在登录节点设置实际 ID 并进入：

```bash
export SOL_H3_SHELL_ID='替换为本次申请返回的ID'
detcli shell open "$SOL_H3_SHELL_ID"
```

此命令按调度器生成的连接方式进入 GPU 容器；不要拿登录节点当计算节点，也不要复用历史端口或旧会话连接信息。进入容器后用 `nvidia-smi` 确认分配到的卡数、型号和进程。后续 Python／torchrun 都在 GPU 容器内运行。

## 3. 代码、权重、环境与输入缓存的位置

| 内容 | 服务器路径 |
|---|---|
| 整理后的代码仓库 | `/labdata2/hwx/Sol-H3-Runtime` |
| 官方源码 | `/labdata2/hwx/Sol-H3-Runtime/official/Sol-H3` |
| 公共 MiniMax-H3 权重 | `/Datasets/public_models/SOL-H3/MiniMax-H3` |
| 公共 FastH3 LoRA | `/Datasets/public_models/SOL-H3/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors` |
| 公共运行环境解释器 | `/Datasets/public_models/SOL-H3/envs/solh3-4090/bin/python` |
| native 研究驱动已使用的解释器 | `/labdata2/hwx/sol-h3/envs/solh3-4090-port-v1/bin/python` |
| native CUDA 工具链 | `/labdata2/hwx/sol-h3/tools/cuda-12.8` |
| 编译器 | `/labdata2/hwx/sol-h3/tools/compiler` |
| 双卡 DiT profiling 模板 | `/labdata2/hwx/sol-h3-8xrtx4090/native-fullopt-4090-20261031T030218Z` |
| 四卡 DiT profiling 模板 | `/labdata2/hwx/sol-h3-8xrtx4090/native-fullopt-g4-4090-20261031T045649Z` |
| 真实条件缓存及共享 native 源码所在目录 | `/labdata2/hwx/sol-h3-8xrtx4090/native-compute-4090-20261030T174243Z` |

仓库地址：[T4t4KAU/Sol-H3-Runtime](https://github.com/T4t4KAU/Sol-H3-Runtime)。官方源码固定上游 revision `781e8341fb3b29d3aea5d50265772e9619340b66`。native 研究模板、缓存和历史 trace 是服务器上的独立实验资产，不包含在仓库克隆中；复现本文使用这些已保留的服务器资产。

公共环境为 Python 3.13.9、PyTorch 2.11.0+cu128。复现下述 native trace 时先使用其已验证的研究解释器，并加载模板中的 `native-deps` 和 `runtime`。不要仅因 Python／Torch 版本相同就直接替换依赖路径。公共目录不原地安装、升级依赖。

## 4. 测试方法：暂存非 DiT 输入，只测 DiT

现有固定请求为文本生成视频、约 5 秒，1344×768、124 帧，seed=0；FastH3 四次主干前向、BF16 计算。真实文本编码结果预先缓存，后续测试无需重新加载和执行大型文本编码器。

缓存文件 `conditioning.pt` 包含：

| 字段 | 内容 |
|---|---|
| `prompt_embeds` | 真实文本编码结果，BF16，形状 `[537, 5120]` |
| `text_token_tags` | 文本 token 标签，int64，形状 `[537]` |
| `prompt` | 与缓存对应的原始提示词 |

具体边界如下：

1. 读取真实条件缓存，准备原生 pipeline 所需的输入；初始视频／音频 latent、timesteps 等按固定请求在计时外构造，暂存在 `state.values` 中。**目前落盘的是文本条件，其余入口状态在内存中暂存，并非都已保存为独立磁盘文件。**
2. 在 `MiniMaxH3DenoiseLoopWrapper.__call__` 入口，将暂存输入放到目标设备，并记录 `latents`、`audio_latents`、`prompt_embeds`、`timesteps` 的形状、dtype 与 SHA256。
3. 等待 CUDA 完成，进行 rank barrier，再同步；从调用原生 denoise loop 前开始计时，直到 loop 返回并完成 CUDA 同步。输入准备和初始放置在计时区间外。
4. 保存视频和音频的最终 latent，检查有限值和输入／输出哈希，通过 `DiTComplete` 正常结束本轮。**VAE、音频解码、视频编码不运行。**

这是完整去噪循环的 DiT 边界，包含其中的 scheduler 更新和 SP 通信；当前显存适配在循环内触发的权重 offload 传输也在墙钟时间中。研究计算算子时在 trace 中另行统计计算 kernel，不能把这个墙钟时间直接称为“纯计算时间”。

native 配置采用 Sol-Attn Triton、原生首步和前两层 dense 策略、QKV 通信融合及 AdaLN 预计算；SP2／SP4，无 FSDP／TP；20 个 block 常驻、30 个 block 固定 CPU offload。QKV 通信 BF16，输出通信 FP8；**这不等于权重或 GEMM 使用 FP8 量化**。SM89 路线固定 eager、关闭 packed input，不能称为所有硬件上所有优化都开启。

缓存只适用于其对应的提示词和输入约束。更换提示词、token 数、分辨率、时长或任务类型时，重新生成真实上游输出并核对模型、LoRA、seed、形状与 dtype；不能保留旧文本条件却声称测量了新请求。比较两个 DiT 实现时，必须复用相同的入口输入。

## 5. 在新目录中复现 DiT profiling

### 5.1 选择匹配模板并准备独立目录

以下均在 GPU 容器内执行。卡数必须与本次分配一致；双卡和四卡分别使用对应模板，不能只改 torchrun 进程数。

```bash
set -e
export SOL_H3_GPUS=2
export B=/labdata2/hwx/sol-h3-8xrtx4090
case "$SOL_H3_GPUS" in
  2) export TEMPLATE="$B/native-fullopt-4090-20261031T030218Z" ;;
  4) export TEMPLATE="$B/native-fullopt-g4-4090-20261031T045649Z" ;;
  *) echo 'Use the matching 2-GPU or 4-GPU driver'; exit 1 ;;
esac
export R="$B/profiles/dit-g${SOL_H3_GPUS}-$(date -u +%Y%m%dT%H%M%SZ)"
export PY=/labdata2/hwx/sol-h3/envs/solh3-4090-port-v1/bin/python
export ASSETS=/Datasets/public_models/SOL-H3
mkdir -p "$R/checkpoints"
for name in native_compute_profile.py compute_observer.py qkv_toggle.py \
            platform_preflight.py wire_preflight.py analyze_native_trace_v2.py; do
  cp "$TEMPLATE/$name" "$R/$name"
done
cp -L "$TEMPLATE/conditioning.pt" "$R/conditioning.pt"
ln -s "$(readlink -f "$TEMPLATE/runtime")" "$R/runtime"
ln -s "$(readlink -f "$TEMPLATE/native-deps")" "$R/native-deps"
ln -s "$ASSETS/MiniMax-H3" "$R/checkpoints/model"
ln -s "$ASSETS/FastH3-4-step-Preview-v1-LoRA" "$R/checkpoints/adapter"
cd "$R"
```

不要重新运行历史 `controller.py`：它绑定了历史 allocation ID，并有旧任务的状态约束。这里直接使用相同 DiT 驱动，在新目录保存本次输出，不覆盖已有证据。

### 5.2 环境与运行前检查

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$R/native-deps:$R/runtime"
export TRITON_CACHE_DIR="$R/triton-cache"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export H3_ULYSSES_COMM_DTYPE=bf16 H3_ULYSSES_OUTPUT_DTYPE=fp8
export H3_SOL_COMPILE_BUCKET_SIZE=0 H3_SOL_PACKED_INPUT=0 H3_MODEL_PROFILE=fast4
export CUDA_HOME=/labdata2/hwx/sol-h3/tools/cuda-12.8
export PATH="/labdata2/hwx/sol-h3/tools/compiler/bin:$B/node08-single-20261026T151723Z/ninja-tool/bin:$PATH"
export LD_LIBRARY_PATH="/labdata2/hwx/sol-h3/tools/compiler/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_P2P_LEVEL
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv > "$R/gpu-identity.txt"
nvidia-smi topo -m > "$R/topology.txt"
"$PY" - <<'PYCHECK'
import hashlib, json, os
from pathlib import Path
import torch
r = Path(os.environ["R"])
n = int(os.environ["SOL_H3_GPUS"])
assert torch.cuda.device_count() == n
assert all("4090" in torch.cuda.get_device_name(i) for i in range(n))
available = int(next(x for x in Path("/proc/meminfo").read_text().splitlines()
                     if x.startswith("MemAvailable:")).split()[1]) * 1024
required = 260_000_000_000 if n == 2 else 460_000_000_000
assert available > required, ("Insufficient host RAM", available, required)
assert (r / "checkpoints/model/modular_model_index.json").is_file()
assert (r / "checkpoints/adapter/dense-datafree/adapter_model.safetensors").is_file()
files = list(r.glob("*.py")) + [r / "conditioning.pt"]
files += list((r / "runtime").rglob("*.py")) + list((r / "native-deps").rglob("*.py"))
manifest = {str(p.relative_to(r)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(r / "execution-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
(r / "memory-before.json").write_text(json.dumps({"available_bytes": available}))
PYCHECK
```

如果主机内存门槛未通过，先等待同节点内存释放或重新选择资源；不要把 CPU OOM 记成 DiT 显存容量结论。检查 `nvidia-smi` 中没有自己的其他计算任务与本次测试抢卡。不要覆盖调度器的 GPU 可见性设置。

### 5.3 预检与 DiT trace 采集

下面各步应顺序执行；任一步失败，停止后续步骤并保留日志。

```bash
set -e
"$PY" "$R/platform_preflight.py" > "$R/platform-preflight.log" 2>&1
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$SOL_H3_GPUS" \
  "$R/wire_preflight.py" > "$R/fp8-wire-preflight.log" 2>&1
"$PY" -u -m torch.distributed.run --standalone --nproc_per_node="$SOL_H3_GPUS" \
  "$R/native_compute_profile.py" > "$R/profile.log" 2>&1
"$PY" "$R/analyze_native_trace_v2.py" > "$R/analysis.log" 2>&1
```

建议在 `tmux` 等持久终端中执行；另一个终端可用 `tail -f "$R/profile.log"` 查看。首次 native kernel 编译和权重准备可能较慢，日志无最终结果时先看 `driver-progress-rank*.json` 所处阶段。

该驱动固定执行一次 warmup、一次带 profiler 的观测轮；rank0 采集 CPU／CUDA trace，所有 rank 执行完整去噪。它验证前后输入及最终 latent 逐位一致，并检查 QKV 融合、FP8 输出通信及 sparse 路径实际执行。

## 6. 输出检查与计时解释

| 文件 | 应检查的内容 |
|---|---|
| `loaded-rank*.json` | 卡数、GPU、软件版本、条件哈希、四次前向、稀疏后端及显存策略 |
| `on-warmup-rank*.json` | warmup 耗时、入口哈希和最终 latent 哈希 |
| `on-measured-rank*.json` | 带 profiler 的 DiT 墙钟 `dit_s`、逐 rank 时长、显存峰值 |
| `profile-verified-rank*.json` | 每个 rank 均 `status=completed`；输入／最终 latent 一致；sparse 执行记录 |
| `on-validated-dispatch-rank*.json` | QKV fused 路径实际调用计数 |
| `observer-rank0.json` | `model_forwards=4`，观察范围及状态 |
| `rank0.trace.json.gz` | CUPTI 原始 trace，可用 Perfetto 查看 |
| `operator-shapes-rank0.json` | 计算算子输入形状 |
| `compute-rank0.json` | 按 DiT 模块归因的计算 kernel 统计 |
| `on-measured-latents.pt` | 去噪后的视频／音频 latent；不是编码视频 |

关键口径：

- `dit_s` 是同步后的各 rank 时长最大值，不是各卡求和或平均。
- 当前 `on-measured` **启用了 profiler**，只能描述该观测轮；warmup 可能包含首次编译和 AdaLN 预计算，也不能充当稳定性能基线。正式比较加速比时，另做相同缓存输入、相同预热与配置、**关闭 profiler** 的独立 benchmark。本文复用的驱动未提供独立 clean-benchmark 开关，不能把两个现有 JSON 改名后作为该数据。
- `compute-rank0.json` 排除 NCCL、memcpy、memset，计算模块占比的分母是该 rank 的计算 kernel 累计时间，**不是完整 DiT 墙钟**。不得把多流、嵌套 scope 或多个 rank 时间直接相加为请求时间。
- 计算 bottleneck 排序应看原生优化实际启用后的 trace，并保留未归因开销。通信、传输可能与计算重叠或干扰，剔除对应事件后不能声称实测了“无传输干扰”的计算时长。
- 输入／输出 latent 哈希一致用于确认观察器未改变该固定请求的结果，不代表新的近似计算策略已经完成感知质量验证。

测试结束并确认日志和输出已持久化后，回到登录节点释放**本次自己申请的**会话：

```bash
detcli shell kill "$SOL_H3_SHELL_ID"
```

接手者首先可直接读取第 3 节两个模板目录中的既有 trace 和验证 JSON；需要新增观测时再申请 GPU，按新 run 目录执行以上步骤。
