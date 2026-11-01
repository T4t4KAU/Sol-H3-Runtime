# SOL-H3：LINs Lab 部署与推理

更新日期：2026-11-01。仓库：[T4t4KAU/Sol-H3-Runtime](https://github.com/T4t4KAU/Sol-H3-Runtime)。以下从已经取得 GPU、进入服务器终端开始。

官方入口支持按资源配置使用 1、2、4 或 8 张 GPU，仍需满足官方显存和算子要求。8×RTX4090 24 GiB 是额外支持的一种适配场景，见第 3 节。

## 1. 设置环境和路径

在同一个终端中依次执行。权重和环境使用公共副本，代码、日志和视频保存在自己的工作目录。

```bash
export SOL_H3_WORK="$HOME/sol-h3-runtime-deploy"
export SOL_H3_ASSETS=/Datasets/public_models/SOL-H3
export SOL_H3_PYTHON="$SOL_H3_ASSETS/envs/solh3-4090/bin/python"
export SOL_H3_MODEL="$SOL_H3_ASSETS/MiniMax-H3"
export SOL_H3_ADAPTER="$SOL_H3_ASSETS/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors"
export SOL_H3_OUTPUT="$SOL_H3_WORK/outputs/video.mp4"
mkdir -p "$SOL_H3_WORK/outputs" "$SOL_H3_WORK/logs"

nvidia-smi
test -x "$SOL_H3_PYTHON"
test -d "$SOL_H3_MODEL"
test -r "$SOL_H3_ADAPTER"
"$SOL_H3_PYTHON" -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("Visible GPUs:", torch.cuda.device_count())'
```

CUDA 应为 `True`，可见 GPU 数量应与分配的资源一致。`/Datasets/public_models/SOL-H3` 整个目录需在当前环境中可见。共享环境使用 Python 3.13.9、PyTorch 2.11.0+cu128；不匹配目标设备时，按 [官方安装说明](https://github.com/T4t4KAU/Sol-H3-Runtime/blob/main/official/Sol-H3/README.md#setup) 准备环境并替换 `SOL_H3_PYTHON`。

## 2. 获取代码

```bash
git clone https://github.com/T4t4KAU/Sol-H3-Runtime.git "$SOL_H3_WORK/repo"
git -C "$SOL_H3_WORK/repo" checkout 03fd1ab7d0b90000a3bfb4027f4bfc4c71ef8c0d
cd "$SOL_H3_WORK/repo"
"$SOL_H3_PYTHON" tools/verify_sources.py
```

已有仓库时跳过克隆，进入对应目录并核对版本；保留已有改动。上述提交是本指南固定使用的运行代码版本。

## 3. 选择运行入口

一般情况使用官方入口，GPU 数量按实际资源改为 `1`、`2`、`4` 或 `8`：

```bash
export SOL_H3_ENTRY=run.sh
export SOL_H3_OFFICIAL_PYTHON="$SOL_H3_PYTHON"
export SOL_H3_OFFICIAL_GPUS=1
```

这里的 `1` 是示例值。显存必须满足官方运行要求；多卡数量本身不保证模型能够装下。下文示例显式使用 `dense` 后端进行基础运行；其他后端按官方支持情况选择。

**仅当使用 8×RTX4090 24 GiB 适配时**，把入口改为以下配置，并运行对应预检：

```bash
export SOL_H3_ENTRY=run_8x4090.sh
"$SOL_H3_PYTHON" tools/preflight_4090.py \
  --model "$SOL_H3_MODEL" --adapter "$SOL_H3_ADAPTER" --gpu
```

两种入口选择一种。启动脚本会管理进程，无需额外套一层 `torchrun`。

## 4. 启动推理并查看日志

使用选定的入口，后台生成一段视频：

```bash
cd "$SOL_H3_WORK/repo"
nohup bash "$SOL_H3_ENTRY" \
  --model "$SOL_H3_MODEL" \
  --adapter "$SOL_H3_ADAPTER" \
  --attention-backend dense \
  --task t2v --duration 5 --seed 0 \
  --prompt 'An adult dancer performs energetic choreography on a brightly lit stage.' \
  --output "$SOL_H3_OUTPUT" \
  > "$SOL_H3_WORK/logs/inference.log" 2>&1 < /dev/null &
echo $! > "$SOL_H3_WORK/logs/inference.pid"
tail -f "$SOL_H3_WORK/logs/inference.log"
```

将 `--prompt` 后的文字替换为自己的提示词。正常输出为 1344×768 视频并带音频。`Ctrl+C` 退出日志查看，不会停止后台生成；后台任务仍要求 GPU 环境保持存活。

## 5. 检查结果

生成结束后执行，确认文件存在且同时包含视频和音频：

```bash
ls -lh "$SOL_H3_OUTPUT"
"$SOL_H3_PYTHON" -c 'import av,sys; c=av.open(sys.argv[1]); assert c.streams.video and c.streams.audio; print("PASS: video and audio streams found"); c.close()' "$SOL_H3_OUTPUT"
```

视频路径为 `$HOME/sol-h3-runtime-deploy/outputs/video.mp4`，修改工作目录时以 `SOL_H3_OUTPUT` 为准。

目录不可访问时先确认公共目录在当前环境可见；运行报错时查看 `$SOL_H3_WORK/logs/inference.log`。不要向共享环境直接安装或升级依赖。

## 当前验证状态

公共权重与环境已准备完成，文件校验、读取权限、依赖导入和不使用 GPU 的启动预检已通过。**尚未用这套公共副本在新的 GPU 环境中完整生成一次视频，因此端到端验收仍未完成。** 使用者完成第 4、5 节后，才算目标环境中的推理通过。
