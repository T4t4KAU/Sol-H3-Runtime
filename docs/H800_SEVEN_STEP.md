# H800 单卡七步常驻服务

七步指[官方 Sol-H3-RTX5090](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-RTX5090) 的两阶段生成路线：H3 四次前向生成 672×384 的草稿，经过 learned ×2 upscaler 和 H3-to-LTX latent adapter，再由 LTX-2.5 三次前向精修，输出 1344×768、121 帧、24 FPS，并保留 H3 原始音频。它不是给单阶段 H3 设置 `num_steps=7`。

本仓库在这条路线外增加单 H800 的常驻服务适配。计算使用 BF16、dense attention，保留固定的 LTX 精修提示词缓存和官方 sigma 调度；H3 阶段使用用户提示词。未宣称复现 RTX5090 稀疏后端的速度。官方和冻结适配源码保持原样。

## 所需资产

在现有 H800 四步环境和 H3 权重之外，准备：

```text
assets/seven-step/
  LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
  LTX-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
  LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
  LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors
  H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors
  H3-to-LTX-Latent-Adapter/config.json
  H3-to-LTX-Latent-Adapter/model.safetensors
  ltx-fixed.pt
```

还需外部源码：Sana `670482d8a857d578ac8a2ea89b052d0fb47badba`、LTX-2 `d151147788a9284cca791edc6ce898007e727fe6`，以及 SHA256 为 `744063b43e0f3eec23e2485cb7c65503069946ca9690906ecb548d7515cb89e2` 的 upscaler Python 源文件。LTX 版本需提供 `TileSizeConfig`；不能随意替换为其他源码目录。资产和源码与代码仓库分开保存，不会在 API 请求中下载。

额外依赖见 `requirements-h800-seven-step.txt`。当前部署将其装在独立 `seven-step-extra` 目录，并通过配置的 `extra_python_path` 载入，不改动已验证的四步环境。`ltx-fixed.pt` 是经过指纹、固定提示词和 tensor 结构校验的离线提示词缓存，服务中不加载 LTX 的文本编码器。

## 启动

在已取得一张 H800 80 GiB 的计算节点上执行。路径均需在该节点可读；配置格式见 `examples/h800-seven-step.json`。

```bash
export SOL_H3_ROOT=/storage/lintaoLab/yansenhan/sol-h3
export SOL_H3_PYTHON="$SOL_H3_ROOT/envs/sol-h3/bin/python"
export SOL_H3_MODEL="$SOL_H3_ROOT/assets/MiniMax-H3"
export SOL_H3_ADAPTER="$SOL_H3_ROOT/assets/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"
unset SOL_H3_API_KEY
cd "$SOL_H3_ROOT/repo"
bash serve.sh \
  --host 0.0.0.0 --port 18080 \
  --backend h800-seven-step \
  --seven-step-config "$SOL_H3_ROOT/deployment/seven-step/config.json" \
  --model "$SOL_H3_MODEL" --adapter "$SOL_H3_ADAPTER" \
  --data-dir "$SOL_H3_ROOT/api-data" --capacity 16
```

由 Slurm 托管时，在批处理脚本中执行同样的命令即可。现有部署也支持在 `deployment/h800.env` 中设置 `SOL_H3_BACKEND=h800-seven-step` 和 `SOL_H3_SEVEN_STEP_CONFIG`，作为未显式指定参数时的默认值。不要在同一张已被服务占用的卡上再启动另一个实例。

`/health` 在模型加载完成前返回 503。就绪后 `backend=h800-seven-step`，`generation_mode=H3(4)+LTX(3)`，所有组件加载次数为 1。API 提交、排队、查询和下载方式沿用 [API 指南](API.md)，无需密钥。

模型只在启动时加载、合并 LoRA。单卡按阶段将 H3 和 LTX 在 CPU/GPU 间切换，各组件对象持续保留；常驻服务并不意味着两套完整模型同时放在 GPU。请求结果包含 `forward_counts={"h3":4,"ltx":3}`、模型身份、峰值显存和阶段时间，任务目录另保存 `seven_step_receipt.json`。

七步和四步是不同生成路径，帧数及计算量不同，不用两者的功能验收耗时直接宣称优化加速比。切回四步使用 `--backend h800-resident`，同时取消环境中的七步默认配置。切换前确认没有运行或排队任务，再重启服务。

## 验收记录

2026-11-03 在单张 H800 80 GiB 上完成连续两条真实 API 请求：实际前向均为 H3=4、LTX=3，输出均为 1344×768、121 帧、24 FPS；视频和音频完整解码通过。工作进程、模型实例及九个组件对象保持一致，各组件加载次数均为 1，FIFO 串行执行正常，未使用 API 密钥。启动预加载约 358 秒，两条功能请求分别约 114 秒、69 秒，提示词和种子不同，不据此推导加速比。详见 `VALIDATION.json`。
