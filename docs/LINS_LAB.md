# SOL-H3 部署与推理

更新日期：2026-11-02。仓库：[T4t4KAU/Sol-H3-Runtime](https://github.com/T4t4KAU/Sol-H3-Runtime)。以下从已经取得 GPU、进入服务器终端开始。

本指南示例使用 FastH3 四步权重，生成约 5 秒、带音频的视频。服务器配置依据 LINs Lab 的设备记录与完整生成验证整理，实际配置以分配到的 GPU 为准。

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
```

`/Datasets/public_models/SOL-H3` 整个目录需在当前环境中可见。共享环境使用 Python 3.13.9、PyTorch 2.11.0+cu128；不匹配目标设备时，按 [官方安装说明](https://github.com/T4t4KAU/Sol-H3-Runtime/blob/main/official/Sol-H3/README.md#setup) 准备环境并替换 `SOL_H3_PYTHON`。

## 2. 放入代码

在本机准备好 [Sol-H3-Runtime 仓库](https://github.com/T4t4KAU/Sol-H3-Runtime) 的完整目录，使用现有文件传输方式上传到服务器的 `$SOL_H3_WORK/repo`。也可在本机下载仓库 ZIP，解压后上传；无需在服务器执行 `git clone`。

上传后，在服务器执行：

```bash
cd "$SOL_H3_WORK/repo"
test -f scripts/run_lins_rtx4090_24g_8gpu.sh
```

已有代码时直接复用，并确认是当前仓库版本。代码传输不包含模型权重和 Python 环境，它们使用第 1 节的公共路径。

## 3. 使用对应的服务器配置

| 服务器配置 | 本仓库的完整推理入口 |
|---|---|
| RTX4090 24 GiB，8 卡 | `scripts/run_lins_rtx4090_24g_8gpu.sh` |
| RTX4090 24 GiB，2 / 4 卡 | 独立引擎现有记录仅覆盖 DiT 阶段，尚未提供完整生成入口 |
| RTX4090 48 GiB，1 卡 | 既有生成记录属于另一条推理路线，尚未整理进本仓库 |

对表中已有完整推理入口的配置，设置相应脚本：

```bash
export SOL_H3_ENTRY=scripts/run_lins_rtx4090_24g_8gpu.sh
```

脚本读取对应服务器配置，自动检查权重、环境和 GPU，再启动推理。无需单独运行预检或设置进程数。其他配置需要完成对应适配与验证后再提供脚本。

## 4. 启动推理并查看日志

使用选定的入口，后台生成一段视频：

```bash
cd "$SOL_H3_WORK/repo"
nohup bash "$SOL_H3_ENTRY" \
  --model "$SOL_H3_MODEL" \
  --adapter "$SOL_H3_ADAPTER" \
  --task t2v --duration 5 --seed 0 \
  --prompt 'An adult dancer performs energetic choreography on a brightly lit stage.' \
  --output "$SOL_H3_OUTPUT" \
  > "$SOL_H3_WORK/logs/inference.log" 2>&1 < /dev/null &
echo $! > "$SOL_H3_WORK/logs/inference.pid"
tail -f "$SOL_H3_WORK/logs/inference.log"
```

将 `--prompt` 后的文字替换为自己的提示词。正常输出为 1344×768 视频并带音频。`Ctrl+C` 退出日志查看，不会停止后台生成；后台任务仍要求 GPU 环境保持存活。

## 5. 检查结果

生成结束后执行，确认文件存在，并完整解码检查视频和音频：

```bash
ls -lh "$SOL_H3_OUTPUT"
"$SOL_H3_PYTHON" - "$SOL_H3_OUTPUT" <<'PY'
import av
import sys

with av.open(sys.argv[1]) as media:
    assert media.streams.video and media.streams.audio, "Missing video or audio stream"
    stream = media.streams.video[0]
    width, height, fps = stream.width, stream.height, stream.average_rate
    video_frames = sum(1 for _ in media.decode(video=0))
with av.open(sys.argv[1]) as media:
    audio_frames = sum(1 for _ in media.decode(audio=0))
assert video_frames > 0 and audio_frames > 0, "Empty video or audio"
print(f"PASS: {width}x{height}, {fps} FPS, {video_frames} video frames, {audio_frames} audio frames")
PY
```

视频路径为 `$HOME/sol-h3-runtime-deploy/outputs/video.mp4`，修改工作目录时以 `SOL_H3_OUTPUT` 为准。

目录不可访问时先确认公共目录在当前环境可见；运行报错时查看 `$SOL_H3_WORK/logs/inference.log`。不要向共享环境直接安装或升级依赖。

## 验证记录

2026-11-02 已按本文流程完成端到端验证：从本机上传源码，使用公共环境和权重，在 `128c256t_768_4090` 资源池的 8×RTX4090 24 GiB 上生成视频。输出为 1344×768、124 帧、24 FPS，视频与音频均完整解码通过。此次使用 768 GiB 主机内存，未重新验证 512 GiB 节点。其他硬件配置仍以各自验证记录为准。
