# SOL-H3 独立推理代码

这是当前研究所用 **8 × RTX 4090 七步推理 baseline** 的独立源码副本。主入口、八卡七步启动脚本和 `h3_runtime/` 从实际测量使用的冻结部署逐文件校验后提取，保持原始内容。

不包含 AutoResearch、VeloQ、研究状态库、GPU 调度器或实验控制器；未合入静态 AdaLN owner 改造，也未合入四项消融实验新增的通信量化适配。原有可选优化实现仍保留，七步启动脚本默认关闭。

## 目录与执行过程

```text
infer.py                         CLI、模型配置、请求计时、输出保存
run_two_stage_8x4090.sh           当前七步 baseline 的启动入口
run_single_stage_8x4090.sh        补充：FastH3 四步入口
run_original_8x4090.sh            补充：原始 H3 四十九步入口
h3_runtime/
  engine.py                      引擎初始化与配置
  staged.py                      分阶段加载、执行与释放
  text_fsdp.py                   文本编码器分片
  fsdp.py, distributed_loading.py H3 权重加载与 FSDP2 分片
  cp_plan.py, ulysses.py          H3 序列并行与可选 packed SP
  two_stage.py, upstream.py       H3 → latent bridge → LTX
  ltx_sp.py, linear_layout.py     LTX 并行与 Linear 布局
  sparse_attention.py            可选稀疏 Attention
  comm_quant.py, relayout.py      可选通信量化与张量重排
  fusion_install.py, fusions.py   可选算子融合
  adaln.py                       原有 AdaLN 预计算实现
  seven_step_profiling.py         七步计时、执行次数与显存记录
  seven_step_output.py            原始视频/音频输出
  third_party/                   已随原代码携带的 Sol-Attn 代码及许可说明
requirements.txt                 原项目依赖声明
external_sources.json            外部运行时源码版本
EXPORT_MANIFEST.json             来源与逐文件 SHA-256
```

请求过程：文本模型加载和文本编码 → H3 加载（含 LoRA 融合）及四步去噪 → latent bridge/upscaler → LTX 加载及三步去噪 → 视频解码，配合 H3 音频生成输出。

固定的七步工作负载是 T2V、5 秒视频：H3 中间分辨率 672×384，最终 1344×768、121 帧、24 fps，音频来自 H3。

## baseline 配置

- 8 张 RTX 4090 24 GiB，FSDP8 + SP8，组件分阶段驻留。
- H3：dense Flash SDPA；LTX：efficient SDPA、full-sequence Linear。
- 保留模型原有 FP32/BF16 混合参数；关闭 TF32、BF16 reduced-precision reduction。
- 使用 dense-datafree FastH3 LoRA；H3 四步 + LTX 三步。
- AdaLN 预计算、kernel fusions、packed SP、compute quantization、parallel VAE 默认关闭；VAE 使用 eager tiled decoding。
- 这里的“原始 baseline”指当前八卡七步系统；`run_original_8x4090.sh` 则是另一个不使用 LoRA 的 49 步模型配置，不能将两者差异直接当作引擎加速比。

## 环境及外部资产

代码已从研究框架中独立出来，但模型权重、Python 环境、Sana/LTX 外部源码和 prompt cache 需要另行提供，没有复制进此目录。

原项目记录的运行环境为 Linux、Python 3.13.9、PyTorch 2.11.0+cu128。依赖版本见 `requirements.txt`；它是原项目的依赖声明，本次没有重新安装或认证全新环境。LTX 另需 `scipy==1.18.1`。优先复用已验证的推理环境。

外部源码固定版本见 `external_sources.json`：

| 来源 | revision |
|---|---|
| Sana | `670482d8a857d578ac8a2ea89b052d0fb47badba` |
| LTX-2 | `d151147788a9284cca791edc6ce898007e727fe6` |
| H3 latent upscaler | `d7c01b9011f2e8439493f6c02c29995a27df276f` |

`h3_runtime/upstream.py` 会从 `--sana-root` 的 `models/minimax_h3/Sol-H3-RTX5090/runtime` 加载 bridge/refiner 实现，从 `--ltx-root` 的 `packages/*/src` 加载 LTX 模块。这些是推理依赖，不依赖原 AutoResearch 仓库。

七步需要：

```text
checkpoints/
  MiniMax-H3-Diffusers/                # H3 全模型，包括文本编码器和 VAE
  FastH3/dense-datafree/adapter_model.safetensors
  H3-to-LTX-Latent-Adapter/
  H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors
  LTX-2.5/
    diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
    loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
    vae/ltx-2.5-video-vae-conv-bf16.safetensors
    vae/ltx-2.5-audio-vae-bf16.safetensors
```

H3 接收用户 prompt；LTX 使用固定 refinement prompt 的已验证 cache。`download_checkpoints.py` 只负责 H3/FastH3 下载，不会准备齐全部七步资产，也不固定模型 revision。精确复现应复用原实验的模型快照及 cache。`UPSTREAM_SOURCE_SNAPSHOT.json` 是上游历史来源资料，其中 B300 的性能数字不是本目录在 RTX 4090 上的新测量。

## 启动七步推理

以下路径需替换成执行机器上的实际位置；脚本会自行启动八个进程，不要再包一层 `torchrun`。

```bash
cd /home/hwx/Documents/codes/Sol-H3-Inference
export SOL_H3_PYTHON=/path/to/sol-h3-env/bin/python
export H3_ASSETS=/path/to/checkpoints
export H3_SOURCES=/path/to/sources
export H3_OUTPUTS=/path/to/output
mkdir -p "$H3_OUTPUTS"

./run_two_stage_8x4090.sh \
  --model "$H3_ASSETS/MiniMax-H3-Diffusers" \
  --adapter "$H3_ASSETS/FastH3/dense-datafree/adapter_model.safetensors" \
  --checkpoints "$H3_ASSETS" \
  --sana-root "$H3_SOURCES/Sana" \
  --ltx-root "$H3_SOURCES/LTX-2" \
  --upscaler-source "$H3_SOURCES/h3-upscaler/nodes/minimax_h3_latent_upscaler_3d.py" \
  --prompt-cache /path/to/cache/ltx-fixed.pt \
  --prompt 'A red ball rolls across a wooden table.' \
  --seed 42 --output "$H3_OUTPUTS/seven.mp4"
```

需要原始计时和输出时，在上述命令中追加：

```bash
--benchmark-dir "$H3_OUTPUTS/benchmark-unique" \
--raw-output-path "$H3_OUTPUTS/raw-unique.pt"
```

每次使用新的输出目录。`pipeline_elapsed_s` 包含请求内加载、预计算（如开启）、去噪、解码和 CPU 原始输出整理；启动、原始文件保存及 MP4 编码单列。各阶段 host wall time 可能嵌套，不能直接相加为完整请求时间。

可用 `SOL_H3_RANK_LAUNCHER="$PWD/tools/seven_step_rank_launcher.py"` 选择原实验所用的进程组启动器；默认使用脚本内的 `torch.distributed.run`。两者均不需要研究框架。

## 本次整理的验证范围

逐文件 SHA-256 校验、Python 语法检查、Shell 语法检查，以及三个启动脚本的 `--help` 检查。没有为本次源码整理重新运行 GPU 推理；本地是否具备模型资产与八卡环境不由这些检查证明。

`EXPORT_MANIFEST.json` 区分冻结推理文件和从原项目补充的启动/依赖文件；`EXPORT_CHECKS.json` 保存本次静态检查结果。原目录及正在等待执行的消融实验保持原位置。
