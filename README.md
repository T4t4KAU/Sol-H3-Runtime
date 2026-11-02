# SOL-H3

基于 NVlabs/Sana 的官方 SOL-H3 独立推理引擎，保留完整官方实现，以官方代码作为默认入口。**额外提供 8×RTX4090 24 GiB 的可选适配**，通过单独的 Bash 脚本调用。

官方源码固定于 `781e8341fb3b29d3aea5d50265772e9619340b66`；82 个文件均与上游 Git blob 一致。适配代码使用已有完整生成验证的冻结版本。模型权重、Python 环境和生成结果另行准备。LINs Lab 的权重统一使用 `/Datasets/public_models/SOL-H3/`；代码、环境与输出仍各自存放。


LINs Lab 的运行步骤见 [部署与推理指南](docs/LINS_LAB.md)，从取得 GPU 后开始。

需要通过 HTTP 提交任务时，见 [API 服务部署](docs/API.md)。服务提供串行请求队列、状态查询和视频下载，复用现有启动脚本。

## LINs Lab 服务器配置

现有完整生成部署对应 `scripts/run_lins_rtx4090_24g_8gpu.sh`，配置文件为 `configs/lins_rtx4090_24g_8gpu.env`。它来自 RTX4090 24 GiB 八卡的冻结部署记录，使用公共环境与权重，并自动检查设备和启动条件。

其他已观察到的服务器配置及其验证范围记录在 `configs/lins_lab_support.json`。官方源码支持的进程数不等同于这些服务器已完成验证的部署配置。

## 官方实现：默认使用

环境安装、权重下载、任务和硬件要求以 [官方 README](official/Sol-H3/README.md) 为准。根目录 `requirements.txt` 引用官方依赖；PyTorch 按官方要求先行安装。

代码仓库：[T4t4KAU/Sol-H3-Runtime](https://github.com/T4t4KAU/Sol-H3-Runtime)。

```bash
git clone https://github.com/T4t4KAU/Sol-H3-Runtime.git
cd Sol-H3-Runtime
```

在匹配的官方环境中执行：

```bash
cd /path/to/Sol-H3-Runtime
export SOL_H3_OFFICIAL_PYTHON=/path/to/official-env/bin/python
export SOL_H3_OFFICIAL_GPUS=8
export SOL_H3_ASSETS=/Datasets/public_models/SOL-H3

python3 tools/verify_sources.py
./run.sh \
  --model "$SOL_H3_ASSETS/MiniMax-H3" \
  --adapter "$SOL_H3_ASSETS/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors" \
  --task t2v --duration 5 \
  --prompt 'An adult dancer performs on a stage.' \
  --output /path/to/output.mp4
```

`run.sh` 通过 `run_official.sh` 调用原样官方 `infer.py`。GPU 数量可设置为 1/2/4/8，单 GPU 需要 `--attention-backend dense`。这保留官方使用方式，不代表所有这些配置都已在 LINs Lab 服务器通过完整生成验证。多卡使用官方 Ulysses 路线，所需显存和算子支持取决于设备；SP 本身不分片模型权重。

启动脚本会处理进程启动，无需再套一层 `torchrun`。其他官方参数原样透传。运行 `./run.sh --help` 查看包装器说明，完整参数见官方 `infer.py --help`。

## 可选适配：RTX4090 24 GiB

我们额外支持 **8 张 RTX4090 24 GiB**，使用独立脚本选择适配实现；启动时自动检查依赖、权重和设备，检查通过后运行：

```bash
export SOL_H3_PYTHON=/Datasets/public_models/SOL-H3/envs/solh3-4090/bin/python
export SOL_H3_ASSETS=/Datasets/public_models/SOL-H3
./scripts/run_lins_rtx4090_24g_8gpu.sh \
  --model "$SOL_H3_ASSETS/MiniMax-H3" \
  --adapter "$SOL_H3_ASSETS/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors" \
  --task t2v --duration 5 --seed 0 \
  --prompt 'An adult dancer performs on a stage.' \
  --output /path/to/output-4090.mp4
```

该入口采用 FSDP8、SP8、分阶段加载及 tiled VAE，使用 FastH3 四次 DiT 前向，输出完整视频和音频。五秒 T2V 为 1344×768、124 帧、24 FPS。LINs Lab 的适配环境也统一放在上述公共目录的 `envs/solh3-4090/`。依赖安装及权重、设备预检见 [4090 适配部署说明](docs/DEPLOY_4090.md)，依赖入口是 `requirements-4090.txt`。

适配入口保留已验证的 dense Flash SDPA、混合参数类型和 eager VAE 配置；未默认开启稀疏注意力、AdaLN 预计算、packed SP、原生 kernel fusions、量化计算/通信或 parallel VAE。它是显存适配配置，其性能不能代表官方全优化配置。

可选入口还包括 `run_original_8x4090.sh`（原版 H3，49 次前向）和 `run_two_stage_8x4090.sh`（H3 四次＋LTX 三次前向，另需外部源码与权重）。

## 代码组织

```text
official/Sol-H3/          完整、原样官方实现和第三方声明
run.sh                   默认官方入口
run_official.sh          官方进程启动包装器
scripts/                 按服务器部署记录整理的启动脚本
configs/                 服务器配置与支持范围记录
requirements.txt         官方依赖入口
adapters/rtx4090/         可选的冻结硬件适配实现
run_8x4090.sh             可选：8×RTX4090 24 GiB 单阶段推理
requirements-4090.txt     可选适配依赖
examples/4090.env.example 可选适配环境变量模板
tools/verify_sources.py   源码完整性检查
tools/preflight_4090.py   可选适配的权重、依赖和八卡预检
docs/ADAPTATION.md        官方与适配的边界
SOURCES.json             上游 revision、Git blob 和源文件 SHA256
VALIDATION.json          本次代码整理的检查记录
```

整个目录可独立复制到服务器，无需原工作区路径或软链接。两套实现有各自的 `h3_runtime` 和依赖环境，入口显式选择对应代码。适配基于不同的历史上游版本，未直接修改所收录的官方源码，详见 [适配边界](docs/ADAPTATION.md)。来源及第三方许可说明见 [NOTICE.md](NOTICE.md)。

## 验证范围

本次检查源码身份、Python/Bash 语法、官方与适配启动器的参数传递和目录迁移。结果见 `VALIDATION.json`。2026-11-02 已从独立目录使用公共环境和权重，在 8×RTX4090 24 GiB（768 GiB 主机内存）上完成四步视频生成和视频/音频完整解码验证。验证范围与记录见 `docs/LINS_LAB.md` 和 `VALIDATION.json`。
