# 官方实现与可选硬件适配的边界

根入口 `run.sh` 调用官方实现；`run_8x4090.sh` 额外提供 8×RTX4090 24 GiB 适配。项目本身不绑定这套硬件。

## 官方源码

`official/Sol-H3/` 原样收录 NVlabs/Sana `sol-engine` 的子目录：

- 上游仓库：<https://github.com/NVlabs/Sana>
- revision：`781e8341fb3b29d3aea5d50265772e9619340b66`
- 原路径：`models/minimax_h3/Sol-H3/`
- 82 个源文件逐一匹配上游 Git blob，另记录 SHA256。

完整保留 README、依赖、下载与推理入口、原生融合/注意力/通信实现、测试及第三方代码声明。没有给官方代码添加自定义 FSDP 或 offload 参数。

## 适配源码

`adapters/rtx4090/` 复用 2026-10-26 导出的推理代码。其运行时与成功运行的冻结部署逐文件一致，继承以下部分：

| 模块 | 八卡职责 |
|---|---|
| `distributed_loading.py`、`fsdp.py` | CPU 权重源、混合 dtype/缓冲区处理及 FSDP2 分片 |
| `text_fsdp.py` | 原生文本模型入口及八卡权重分片 |
| `cp_plan.py`、`ulysses.py` | packed 序列的 SP8，处理不等长切分 |
| `staged.py` | 文本编码、H3 去噪、解码之间的组件加载与释放 |
| `engine.py`、`infer.py` | 完整生成、路径/参数检查及输出保存 |
| `run_single_stage_8x4090.sh` | 固定八卡运行策略 |

FSDP8 和 SP8 使用同一组八个 rank，不需要 64 张卡。4090 适配入口以组件分阶段驻留适配 24 GiB 显存，没有把 CPU checkpoint 组装误称为去噪时 layerwise offload。

该适配有自己的历史上游导入版本；它不是把 FSDP 改动直接补到当前 `781e...` 官方目录的结果。两份实现并置保留，以各自源文件和依赖身份为准。逐文件变化列表见 `source-comparison.json`；差异包含历史版本演进，不能作为直接可应用的补丁。

未合入原研究工作区中的未提交改动。没有收录 AutoResearch 控制器、研究状态库、服务器认证信息、模型、环境或生成视频。

## 4090 适配的配置与可选代码

八卡单阶段默认使用 FastH3 dense-datafree LoRA 和四次 DiT 前向，支持完整视频/音频输出。它使用 dense Flash SDPA、FSDP8、SP8、混合参数 dtype、分阶段加载与 eager tiled VAE；关闭 TF32/BF16 reduced-precision reduction。

官方优化模块仍保存在源码中，但 4090 适配 launcher 关闭 AdaLN 预计算、kernel fusions、packed SP、量化计算/通信和 parallel VAE。当前代码整理不构成这些可选优化在八卡上的新验收。

两阶段入口及原版 49 前向入口原样保留，前者另需固定的外部 Sana/LTX/upscaler 源码和额外权重。4090 单阶段四步入口没有这些依赖。

## 本次检查与 GPU 验证

本次检查源码身份、可独立复制的目录、Python/Bash 语法、启动器选择和参数传递。测试用替代解释器捕获启动命令，不分配 GPU、不执行模型。

八卡生成能力的既有验证记录随原始 `EXPORT_MANIFEST.json`、`EXPORT_CHECKS.json` 和适配说明保留；这些历史记录不是本次新 GPU 实测。当前 `VALIDATION.json` 会明确区分本次 CPU 检查与未执行的 GPU 检查。

`tools/verify_sources.py` 校验所收录源码是否改变；校验通过不等于 GPU 数值正确性通过。部署到新的环境后，仍应完成一次目标配置的真实生成检查。
