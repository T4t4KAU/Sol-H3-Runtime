# SOL-H3 API 服务

在 GPU 服务器上运行 `serve.sh` 即可启动 HTTP 推理服务。服务接收生成请求、按 FIFO 顺序排队执行，提供状态查询、日志和视频下载。一次执行一个任务，每个任务使用对应推理入口的整组 GPU。以下命令均在服务器的 GPU 环境内执行。

H800 单卡入口默认使用常驻推理后端：启动时加载模型、合并 LoRA 并预计算 AdaLN，后续请求复用模型，不再为每个任务启动推理进程。DiT 与 VAE 保留在 GPU，文本编码器保留在 CPU，编码时按层上卡。其他入口目前沿用逐任务启动进程的兼容后端。

## 1. 准备环境

先完成 [LINs Lab 部署指南](LINS_LAB.md) 第 1–2 节，在已取得 GPU 的环境中准备代码、公共权重、公共 Python 环境，并设置 `SOL_H3_WORK`、`SOL_H3_PYTHON`、`SOL_H3_MODEL`、`SOL_H3_ADAPTER` 和 `SOL_H3_ENTRY`。

API 使用 Python 标准库，无需向公共环境安装额外依赖。下述命令沿用已验证的服务器入口；其他配置通过 `--entry` 选择相应启动脚本，默认入口为仓库的官方 `run.sh`。

## 2. 启动服务

在同一个 GPU 环境的终端执行：

```bash
cd "$SOL_H3_WORK/repo"
unset SOL_H3_API_KEY
export SOL_H3_API_DATA="$SOL_H3_WORK/api-data"

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

`--capacity 16` 表示最多接收 16 个尚未结束的任务，包含正在运行的一个任务。超过容量时返回 HTTP 429。默认不设置密钥，调用时无需 Authorization 请求头。

默认监听 GPU 服务器（或 GPU 容器）的 `127.0.0.1:8000`。需要监听服务器其他网卡时，将地址改为 `0.0.0.0`，同样无需设置密钥。容器的端口发布方式由服务器运行环境决定，不影响 API 的启动方式。

```bash
export SOL_H3_API_URL=http://127.0.0.1:8000
curl -fsS "$SOL_H3_API_URL/health"
```

上面的健康检查也在 GPU 服务器内执行。H800 常驻后端在加载模型期间返回 HTTP 503 与 `status=loading`，加载完成后返回 HTTP 200 与 `status=ok`；初始化失败则返回 HTTP 503 与错误信息。响应包含工作进程 PID、模型实例 ID、加载次数和已完成请求数。兼容后端的健康检查只表示 API 工作线程可用，GPU 检查仍在每次任务启动时执行。

如需自行启用鉴权，可在启动前设置 `SOL_H3_API_KEY`；此时请求需附带 `Authorization: Bearer <密钥>`。本文命令默认不启用鉴权。

## 3. 提交任务

```bash
JOB_JSON=$(curl -fsS "$SOL_H3_API_URL/v1/jobs" \
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
curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID"

curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID/log"
```

任务状态为 `queued`、`running`、`completed` 或 `failed`。`failed` 的具体原因见返回的 `error` 和任务日志；日志接口返回最后 64 KiB。`completed` 表示后端成功完成本次请求且生成了非空视频，完整媒体解码可按部署指南第 4 节检查。

状态变为 `completed` 后下载：

```bash
curl -fsS "$SOL_H3_API_URL/v1/jobs/$JOB_ID/video" \
  -o "$SOL_H3_WORK/outputs/$JOB_ID.mp4"
```

视频未就绪时返回 HTTP 409。任务元数据、完整日志和视频保存在 `$SOL_H3_API_DATA/<任务ID>/`，多个请求的文件相互独立。

## 5. 停止与恢复

```bash
kill -TERM "$(cat "$SOL_H3_WORK/logs/api.pid")"
```

停止服务会终止当前推理任务，未开始的请求保留在磁盘队列中。使用相同 `--data-dir` 重启后，排队请求按原顺序继续，已中断的任务标记为 `failed`，由客户端决定是否重新提交。重启时沿用相同入口和权重配置。

一个数据目录只允许一个服务进程。推理子进程也持有目录锁，服务异常退出后，若原推理进程仍存活，新的服务会拒绝争用；应先等待其退出或停止原进程组。GPU 资源分配结束或容器退出时，API 随之停止；`nohup` 不会延长 GPU 分配。

当前没有自动清理输出；需要时清理已结束任务的目录。不要删除正在排队或运行的任务，也不要在同一组 GPU 上同时启动多个不同数据目录的服务。

## 验证范围

已用 CPU 替代推理进程测试 HTTP 提交、鉴权、参数校验、队列容量、串行执行、状态查询、下载、失败记录和重启恢复。2026-11-02 已在 8×RTX4090 24 GiB 上完成真实 API 请求验收：鉴权提交、状态查询、HTTP 视频下载、视频和音频完整解码均通过；本机转发通道访问也已验证。输出为 1344×768、124 帧、24 FPS。

## H800 80 GiB 单卡

在 Slurm 分配的一张 H800 上使用 `scripts/run_h800_80g_1gpu.sh`。先设置当前服务器上实际的 `SOL_H3_PYTHON`、`SOL_H3_MODEL`、`SOL_H3_ADAPTER` 和 `SOL_H3_API_DATA`；这些路径与 LINs Lab 的公共路径可以不同。模型必须完整位于计算节点可见的存储上，不能在请求处理中下载权重。

```bash
cd /path/to/Sol-H3-Runtime
unset SOL_H3_API_KEY
bash serve.sh \
  --host 0.0.0.0 --port 18080 \
  --backend h800-resident \
  --entry scripts/run_h800_80g_1gpu.sh \
  --model "$SOL_H3_MODEL" --adapter "$SOL_H3_ADAPTER" \
  --data-dir "$SOL_H3_API_DATA" --capacity 16
```

使用 `sbatch` 托管此命令可在退出终端后继续提供服务；申请 `--gres=gpu:1`，并根据站点配置设置 account、partition 和运行时限。常驻后端在启动时完成模型加载和 AdaLN 预计算，保持单卡 dense BF16 路径；不调用八卡启动器，不要求 CUDA13 的 BSA 组件。

客户端在服务器网络内访问 `http://计算节点名:18080`，提交、排队、查询和下载接口与上文一致，无需密钥。GPU 分配结束时服务停止；模型加载完成后 `/health` 才返回 HTTP 200，实际生成是否成功以任务结果及媒体检查为准。

2026-11-03 已在一张 H800 80 GiB 上通过真实 API 请求：异步提交、状态查询、鉴权及 HTTP 视频下载均正常，输出的 1344×768、124 帧、24 FPS 视频及音频完整解码通过。此配置使用 FastH3 四次去噪和分阶段加载；本次为功能验收，不是独立 DiT benchmark。

随后已切换为免密钥服务，验证无 Authorization 请求头的状态查询、日志及已有视频下载均返回 HTTP 200；下载文件 SHA256 与原视频一致。免密钥提交、执行和下载流程另通过 CPU 模拟推理测试。

`--backend auto` 为默认值：选择单卡 H800 入口时使用 `h800-resident`，其他入口使用 `subprocess`。常驻后端不执行 `--entry` 脚本，而是在 API 工作线程中使用同一 H800 推理配置；`--launch-arg` 只供兼容后端使用。服务仍按 FIFO 串行处理请求，任务结果额外记录工作进程 PID、模型实例 ID、加载次数与请求耗时。

2026-11-03 已完成 H800 常驻后端的连续两请求验收：同一工作进程、同一模型实例和组件对象，DiT 权重存储地址保持不变，各组件加载次数始终为 1；两段 1344×768、124 帧、24 FPS 视频及音频完整解码通过。模型启动预加载约 276 秒，之后处理请求时不重新加载。

## H800 官方七步生成路线

七步为 H3 四次前向加 LTX 三次前向，经 learned ×2 upscaler 和 latent adapter 输出 1344×768、121 帧、24 FPS，保留 H3 音频。单卡 H800 的常驻后端为 `h800-seven-step`，需准备额外的 LTX 权重、固定提示词缓存及匹配源码，启动步骤见 [H800 七步常驻服务](H800_SEVEN_STEP.md)。API 的提交、排队、查询和下载接口保持一致，无需密钥。

H3 和 LTX 组件在启动时各加载一次，请求执行时按阶段在 CPU/GPU 间切换；模型对象持续保留。该 H800 适配使用 dense attention，不代表官方 RTX5090 稀疏后端的速度。请求结果中的 `forward_counts` 校验 H3=4、LTX=3，不能只凭配置参数将单阶段七次迭代称为官方七步。

2026-11-03 已在 H800 七步常驻后端完成连续两请求验收：两条均观测 H3=4、LTX=3，输出均为 1344×768、121 帧、24 FPS，视频音频完整解码通过，九个组件加载次数均保持为 1。四步后端仍独立保留。
