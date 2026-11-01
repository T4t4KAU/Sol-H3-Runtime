# 可选适配：在 8×RTX4090 24 GiB 上部署

本包默认使用官方实现。仅需适配这套显存受限硬件时，使用本文的独立 Bash 入口 `run_8x4090.sh`。

## 1 复制代码并进入 GPU 环境

在当前账号可写、能挂载进 GPU 容器的共享目录中克隆仓库：

```bash
export SOL_H3_BASE=/path/to/shared/sol-h3-runtime-deploy
mkdir -p "$SOL_H3_BASE"
git clone https://github.com/T4t4KAU/Sol-H3-Runtime.git "$SOL_H3_BASE/repo"
```

然后进入自己的八卡 GPU allocation。在 GPU 环境中执行：

```bash
export SOL_H3_BASE=/path/to/shared/sol-h3-runtime-deploy
cd "$SOL_H3_BASE/repo"
nvidia-smi
python3 tools/verify_sources.py
```

代码不包含调度器申请逻辑。登录节点能访问目录，不代表已经获得 GPU。

## 2 配置 Python 环境

LINs Lab 已有共享推理环境位于公共模型目录中，显式指定其解释器：

```bash
export SOL_H3_PYTHON=/Datasets/public_models/SOL-H3/envs/solh3-4090/bin/python
"$SOL_H3_PYTHON" -c 'import torch,diffusers,transformers; print(torch.__version__,diffusers.__version__,transformers.__version__)'
```

该共享环境同时携带公共 CPython 基础运行时，整个 `/Datasets/public_models/SOL-H3/` 需在 GPU 容器内以相同路径可见。使用者可直接运行；依赖变更另建自己的环境。

需要新建环境时，在代码目录运行：

```bash
python3.13 -m venv /path/to/sol-h3-4090-env
source /path/to/sol-h3-4090-env/bin/activate
python -m pip install -r requirements-4090.txt
python -m pip check
export SOL_H3_PYTHON=/path/to/sol-h3-4090-env/bin/python
```

依赖入口保留冻结版本的具体约束，Diffusers 固定于 `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`。完整新机器安装需在目标服务器检查依赖解析和 CUDA 驱动兼容性；本次没有重新安装认证环境。

模型包括大型 DiT、文本编码器和 VAE。FSDP 降低单卡模型驻留，但 CPU 组装/加载仍需要足够主机内存。模型权重和环境放在源码目录外。

## 3 准备权重

只运行单阶段 FastH3，需要完整 MiniMax-H3 模型及 dense-datafree adapter。LINs Lab 统一放在 `/Datasets/public_models/SOL-H3/`；已有完整权重时直接复用，缺少时下载：

```bash
export SOL_H3_ASSETS=/Datasets/public_models/SOL-H3
mkdir -p "$SOL_H3_ASSETS"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$SOL_H3_PYTHON" official/Sol-H3/download_checkpoints.py --output-dir "$SOL_H3_ASSETS"
```

下载前需已具备模型访问权限。共享目录需对使用者可读，并在 GPU 容器中可见。已有有效快照时直接复用；下载脚本不固定模型 revision，精确复现应使用已有模型快照与文件哈希。

```text
/Datasets/public_models/SOL-H3/
  MiniMax-H3/
    modular_model_index.json    # 或 model_index.json
    transformer/
    text_encoder/
    tokenizer/
    processor/
    scheduler/
    audio_scheduler/
    vae/
    audio_vae/
  FastH3-4-step-Preview-v1-LoRA/
    dense-datafree/adapter_model.safetensors
```

若已有目录名为 `MiniMax-H3-Diffusers`、`FastH3`，直接在 `--model`、`--adapter` 中使用实际路径，无需改名。

## 4 运行生成

```bash
export SOL_H3_OUTPUTS=/path/to/outputs
mkdir -p "$SOL_H3_OUTPUTS"
./run_8x4090.sh \
  --model "$SOL_H3_ASSETS/MiniMax-H3" \
  --adapter "$SOL_H3_ASSETS/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors" \
  --task t2v --duration 5 --seed 0 \
  --prompt 'An adult dancer performs energetic choreography on a stage.' \
  --output "$SOL_H3_OUTPUTS/fasth3.mp4"
```

运行脚本自动检查源码、依赖、权重目录和设备；检查失败会退出，通过后才启动模型。完成生成并检查 MP4 的音视频流后，才算本机部署运行成功。

```bash
ffprobe -v error \
  -show_entries stream=codec_type,width,height,r_frame_rate \
  -show_entries format=duration -of json "$SOL_H3_OUTPUTS/fasth3.mp4"
```

推理通过 PyAV 编码，`ffprobe` 用于检查媒体；系统 `ffmpeg`/`ffprobe` 可以另行安装。默认 launcher 开启离线模式，务必先把完整模型下载到本地。

## 5 常见启动错误

| 现象 | 处理 |
|---|---|
| 要求八张可见卡 | 检查 allocation、可见设备和使用的入口；不要再套 `torchrun` |
| 加载了官方目录但发生 OOM | 八卡 24 GiB 使用根目录 `run_8x4090.sh`，它选择 FSDP 适配实现 |
| 缺少 tokenizer / VAE / 文本编码器 | 提供完整模型目录，不能只传 DiT 权重目录 |
| 模型路径无法自动识别 | 显式传 `--model` 和 `--adapter`，避免同一根目录下多个别名产生歧义 |
| 导入到错误版本的 torch 或 diffusers | 使用 `SOL_H3_PYTHON` 指定目标环境，检查实际解释器与包路径 |
| 尝试增加单卡显存却仍失败 | 不关闭默认 FSDP8；八卡默认策略与单卡/副本策略不同 |

默认八卡入口的密集注意力和关闭优化策略是冻结验证的一部分。要试其他优化，应单独配置并做数值/显存验证，不将其当成本包已验收的默认状态。
