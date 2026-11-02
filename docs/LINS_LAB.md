# SOL-H3 部署与推理

更新日期：2026-11-02。仓库：[T4t4KAU/Sol-H3-Runtime](https://github.com/T4t4KAU/Sol-H3-Runtime)。以下从已经取得 GPU、进入服务器终端开始。

本指南示例使用 FastH3 四步权重，生成约 5 秒、带音频的视频。服务器配置依据 LINs Lab 的设备记录与完整生成验证整理，实际配置以分配到的 GPU 为准。

## 1. 设置环境和路径

在同一个终端中依次执行。权重和环境使用公共副本，代码、日志和视频保存在自己的工作目录。以下默认仓库已位于 `$SOL_H3_WORK/repo`。

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

## 2. 使用对应的服务器配置

| 服务器配置 | 本仓库的完整推理入口 |
|---|---|
| RTX4090 24 GiB，8 卡 | `scripts/run_lins_rtx4090_24g_8gpu.sh` |

对表中已有完整推理入口的配置，设置相应脚本：

```bash
export SOL_H3_ENTRY=scripts/run_lins_rtx4090_24g_8gpu.sh
```

脚本读取对应服务器配置，自动检查权重、环境和 GPU，再启动推理。无需单独运行预检或设置进程数。其他配置需要完成对应适配与验证后再提供脚本。

## 3. 启动推理并查看日志

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

## 4. 检查结果

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

## 5. 启动和使用 API 服务

以下命令均在 GPU 服务器／容器内执行。API 与第 3 节的单次生成使用同一套环境、权重和服务器配置；完成第 1–2 节后，可以直接选择 API 方式运行。

服务接收请求后按顺序生成视频，一次执行一个任务。当前 API 进程常驻，每个任务会重新加载模型。无需安装额外的 API 依赖。

### 5.1 启动服务

在同一个终端中执行。首次启动会生成密钥，后续启动复用同一个密钥文件：

```bash
cd "$SOL_H3_WORK/repo"
test -f serve.sh
export SOL_H3_API_DATA="$SOL_H3_WORK/api-data"
export SOL_H3_API_URL=http://127.0.0.1:8000

if [ ! -s "$SOL_H3_WORK/api.key" ]; then
  (umask 077; "$SOL_H3_PYTHON" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$SOL_H3_WORK/api.key")
fi
export SOL_H3_API_KEY="$(cat "$SOL_H3_WORK/api.key")"

nohup bash serve.sh \
  --host 127.0.0.1 --port 8000 \
  --entry "$SOL_H3_ENTRY" \
  --model "$SOL_H3_MODEL" \
  --adapter "$SOL_H3_ADAPTER" \
  --data-dir "$SOL_H3_API_DATA" \
  --capacity 16 \
  > "$SOL_H3_WORK/logs/api.log" 2>&1 < /dev/null &
echo $! > "$SOL_H3_WORK/logs/api.pid"
```

服务监听当前 GPU 环境的 `127.0.0.1:8000`。`--capacity 16` 表示最多接收 16 个尚未结束的任务，包含正在运行的任务；队列满时返回 HTTP 429。不要在同一组 GPU 上同时运行第 3 节的命令和 API 推理，或重复启动多个服务实例。

需要监听服务器其他网卡时，将 `--host` 改为 `0.0.0.0`，保留密钥设置；容器端口的发布方式按服务器运行环境配置。

查看服务日志和健康状态：

```bash
tail -n 50 "$SOL_H3_WORK/logs/api.log"
curl -fsS "$SOL_H3_API_URL/health"
```

服务启动需要短暂等待；连接失败时先查看日志，再重试健康检查。正常返回包含 `"status": "ok"`。这表示 API 可响应，权重和 GPU 检查会在任务启动时由推理脚本自动执行。

### 5.2 提交生成任务

```bash
JOB_JSON=$(curl -fsS "$SOL_H3_API_URL/v1/jobs" \
  -H "Authorization: Bearer $SOL_H3_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An adult dancer performs energetic choreography on a brightly lit stage.","duration":5,"seed":0}')
printf '%s\n' "$JOB_JSON"
JOB_ID=$(printf '%s' "$JOB_JSON" | "$SOL_H3_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["id"])')
```

成功提交返回 HTTP 202 和任务 ID，生成在后台进行。将 JSON 中的 `prompt` 替换为自己的提示词。

| 参数 | 说明 |
|---|---|
| `prompt` | 必填，1–4000 个字符 |
| `duration` | 当前 API 只开放 5 秒，默认 `5` |
| `seed` | 0–2147483647 的整数，默认 `0` |

### 5.3 查询状态和日志

```bash
curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID" \
  -H "Authorization: Bearer $SOL_H3_API_KEY"

curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID/log" \
  -H "Authorization: Bearer $SOL_H3_API_KEY"
```

任务依次处于 `queued`（排队）、`running`（生成中），最终为 `completed`（完成）或 `failed`（失败）。重复查询状态，等到 `completed` 后再下载。失败时查看返回的 `error` 和任务日志；日志接口返回最后 64 KiB，完整日志保存在 `$SOL_H3_API_DATA/$JOB_ID/inference.log`。

### 5.4 下载和检查视频

任务完成后执行：

```bash
export SOL_H3_OUTPUT="$SOL_H3_WORK/outputs/$JOB_ID.mp4"
curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID/video" \
  -H "Authorization: Bearer $SOL_H3_API_KEY" \
  -o "$SOL_H3_OUTPUT"
ls -lh "$SOL_H3_OUTPUT"
```

视频下载到服务器的 `$SOL_H3_WORK/outputs/`。此时可直接复用第 4 节的解码命令检查结果。生成尚未完成时，下载接口返回 HTTP 409。

每个任务的元数据、原始视频和完整日志均保存在 `$SOL_H3_API_DATA/$JOB_ID/`，不同任务的输出相互独立。

### 5.5 停止和恢复服务

```bash
kill -TERM "$(cat "$SOL_H3_WORK/logs/api.pid")"
```

停止服务会终止当前推理，未开始的任务仍保存在队列中。重新执行第 5.1 节，使用相同的数据目录、权重配置和密钥文件后，排队任务会继续执行；被中断的任务标记为 `failed`，需要重新提交。

换一个服务器终端使用已运行的服务时，先重新执行第 1、2 节的环境变量设置，再执行：

```bash
export SOL_H3_API_DATA="$SOL_H3_WORK/api-data"
export SOL_H3_API_URL=http://127.0.0.1:8000
export SOL_H3_API_KEY="$(cat "$SOL_H3_WORK/api.key")"
```

随后使用已有的任务 ID 查询或下载，不要再次执行启动命令。GPU 分配结束或容器退出时，服务也会停止，`nohup` 不会延长 GPU 分配。
