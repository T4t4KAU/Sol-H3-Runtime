# SOL-H3 API 服务

API 复用仓库现有推理脚本，支持提交生成任务、查询状态、查看日志和下载视频。请求按 FIFO 顺序排队，同一服务一次执行一个任务，每个任务使用启动脚本对应的整组 GPU。

当前版本每个任务都会重新加载模型，耗时包含加载、生成和编码。它提供异步服务接口；模型常驻与跨请求复用尚未实现。

## 1. 准备环境

先完成 [LINs Lab 部署指南](LINS_LAB.md) 第 1–3 节，在已取得 GPU 的环境中准备代码、公共权重、公共 Python 环境，并设置 `SOL_H3_WORK`、`SOL_H3_PYTHON`、`SOL_H3_MODEL`、`SOL_H3_ADAPTER` 和 `SOL_H3_ENTRY`。

API 使用 Python 标准库，无需向公共环境安装额外依赖。下述命令沿用已验证的服务器入口；其他配置通过 `--entry` 选择相应启动脚本，默认入口为仓库的官方 `run.sh`。

## 2. 启动服务

在同一个 GPU 环境的终端执行：

```bash
cd "$SOL_H3_WORK/repo"
export SOL_H3_API_KEY="$("$SOL_H3_PYTHON" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SOL_H3_API_DATA="$SOL_H3_WORK/api-data"

nohup "$SOL_H3_PYTHON" tools/serve_api.py \
  --host 127.0.0.1 --port 8000 \
  --entry "$SOL_H3_ENTRY" \
  --model "$SOL_H3_MODEL" \
  --adapter "$SOL_H3_ADAPTER" \
  --data-dir "$SOL_H3_API_DATA" \
  --capacity 16 \
  > "$SOL_H3_WORK/logs/api.log" 2>&1 < /dev/null &
echo $! > "$SOL_H3_WORK/logs/api.pid"
```

`--capacity 16` 表示最多接收 16 个尚未结束的任务，包含正在运行的一个任务。超过容量时返回 HTTP 429。请保留 `SOL_H3_API_KEY`，后续请求使用相同值；它不会被写入任务记录。

默认仅监听本机。需要其他机器访问时，将监听地址改为 `0.0.0.0`，并确保 GPU 环境的端口可达；该模式必须设置 `SOL_H3_API_KEY`。跨公网访问应通过提供 HTTPS 的反向代理转发。

```bash
export SOL_H3_API_URL=http://127.0.0.1:8000
curl -fsS "$SOL_H3_API_URL/health"
```

`/health` 只表示 API 进程可响应。任务启动时仍会执行推理脚本自带的环境、权重和 GPU 检查。

## 3. 提交任务

```bash
JOB_JSON=$(curl -fsS "$SOL_H3_API_URL/v1/jobs" \
  -H "Authorization: Bearer $SOL_H3_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An adult dancer performs energetic choreography on a brightly lit stage.","duration":5,"seed":0}')
printf '%s\n' "$JOB_JSON"
JOB_ID=$(printf '%s' "$JOB_JSON" | "$SOL_H3_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["id"])')
```

成功提交返回 HTTP 202 和任务 ID。支持的字段：

| 字段 | 含义 |
|---|---|
| `prompt` | 必填，1–4000 个字符 |
| `duration` | 当前服务仅开放已验证的 5 秒，默认 `5` |
| `seed` | 0–2147483647 的整数，默认 `0` |

客户端不能修改执行脚本、模型目录或输出路径。这些由服务启动配置决定。

## 4. 查询状态与下载

```bash
curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID" \
  -H "Authorization: Bearer $SOL_H3_API_KEY"

curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID/log" \
  -H "Authorization: Bearer $SOL_H3_API_KEY"
```

任务状态为 `queued`、`running`、`completed` 或 `failed`。`failed` 的具体原因见返回的 `error` 和任务日志；日志接口返回最后 64 KiB。`completed` 表示推理进程成功退出且生成了非空视频，完整媒体解码可按部署指南第 5 节检查。

状态变为 `completed` 后下载：

```bash
curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID/video" \
  -H "Authorization: Bearer $SOL_H3_API_KEY" \
  -o "$SOL_H3_WORK/outputs/$JOB_ID.mp4"
```

视频未就绪时返回 HTTP 409。任务元数据、完整日志和视频保存在 `$SOL_H3_API_DATA/<任务ID>/`，多个请求的文件相互独立。

## 5. 停止与恢复

```bash
kill -TERM "$(cat "$SOL_H3_WORK/logs/api.pid")"
```

停止服务会终止当前推理任务，未开始的请求保留在磁盘队列中。使用相同 `--data-dir` 重启后，排队请求按原顺序继续，已中断的任务标记为 `failed`，由客户端决定是否重新提交。重启时沿用相同入口、权重配置和 API 密钥。

一个数据目录只允许一个服务进程。推理子进程也持有目录锁，服务异常退出后，若原推理进程仍存活，新的服务会拒绝争用；应先等待其退出或停止原进程组。GPU 资源分配结束或容器退出时，API 随之停止；`nohup` 不会延长 GPU 分配。

当前没有自动清理输出；需要时清理已结束任务的目录。不要删除正在排队或运行的任务，也不要在同一组 GPU 上同时启动多个不同数据目录的服务。

## 验证范围

已用 CPU 替代推理进程测试 HTTP 提交、鉴权、参数校验、队列容量、串行执行、状态查询、下载、失败记录和重启恢复。2026-11-02 已在 8×RTX4090 24 GiB 上完成真实 API 请求验收：鉴权提交、状态查询、HTTP 视频下载、视频和音频完整解码均通过；本机转发通道访问也已验证。输出为 1344×768、124 帧、24 FPS。
