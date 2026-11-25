# Laya-MLX

**开放权重的类型化决策模型，在 Apple Silicon 上原生运行。**

[English](https://github.com/mizorewww/laya-mlx/blob/main/README.md) · [使用参考](https://github.com/mizorewww/laya-mlx/blob/main/docs/USAGE.md) · [基准测试](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) · [Hugging Face 权重](https://huggingface.co/aac6fef/laya-mlx) · [更新日志](https://github.com/mizorewww/laya-mlx/blob/main/CHANGELOG.md)

Laya 专门回答有固定答案范围的问题：从几个选项里挑一个、按评分标准打分、判断一句话真不真。
它只做一次双向前向计算，直接给出校准后的概率，不逐个生成 token，也不需要解析输出的 JSON。
本项目用 [MLX](https://github.com/ml-explore/mlx) 运行 [Laya](https://github.com/NandhaKishorM/laya)
原版权重，不依赖 PyTorch、Transformers 或任何云端 API。

M3 Max、FP16 下，一个简短的英文问题中位延迟 **13.4 ms**，多语言模型 **7.4 ms**。

![Laya MLX 玩贪吃蛇：真实决策，原速录制](https://raw.githubusercontent.com/mizorewww/laya-mlx/main/docs/assets/snake-demo.gif)

## 快速开始

需要 Apple Silicon、macOS 14+ 和 Python 3.11+。

```bash
pip install laya-mlx
```

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "Payments, invoices, refunds, and duplicate charges.",
                "technical": "Broken features, errors, and product troubleshooting.",
                "sales": "Pricing, upgrades, and new purchases.",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
)
print(result["answers"]["department"]["choice"])  # billing
```

像上面这样给每个选项写上说明，效果远好于只给标签名。首次加载会下载模型，之后的推理完全在本地进行。

## 问题类型

| 类型 | 你提供 | 你得到 |
|---|---|---|
| `choice` | 若干选项：列表，或「标签 → 说明」字典 | 每个选项的概率 |
| `score` | 有序的评分档位 | 每档的概率和期望档位 |
| `noul` | 一个是非命题 | 为真的概率 |

输入状态可以是文本、类 JSON 字典或对话列表。每个答案都附带校准概率和置信度；
在新任务上，置信度高不代表一定答对。

## 模型

| 预转换 MLX 版（FP16） | 原版 | 参数量 | 上下文 | 用途 |
|---|---|---:|---:|---|
| [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) | `convaiinnovations/laya` | 421M | 512 | 英文 |
| [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) | `convaiinnovations/laya-multilingual` | 322M | 1,024 | 多语言输入 |
| [`aac6fef/laya-typed-decisions-mlx`](https://huggingface.co/aac6fef/laya-typed-decisions-mlx) | `convaiinnovations/laya-typed-decisions` | 421M | 1,024 | 上游 typed-decisions 工作流 |

`laya.load` 两列的 ID 都能用。上下文长度把指令、选项和状态算在一起。
每个发布的模型都带有模型卡、验证结果、来源说明和校验和（[固定版本与哈希](https://github.com/mizorewww/laya-mlx/blob/main/benchmarks/results/hub-publication.json)）。

## 其他功能

详见[使用参考](https://github.com/mizorewww/laya-mlx/blob/main/docs/USAGE.md)：

- **语言路由**：`Router()` 自动识别语言、选择模型，默认最多常驻两个；也可以注册自己的模型。
- **批量与长文档**：`predict_batch` 对问题结构相同的请求共用前向计算；长文本扫描会保留答案出自哪一段。
- **置信度门槛**：`min_confidence=` 把低置信度答案标记出来供人工复核，答案本身不变。
- **超多选项**：`predict_shortlist`（先用向量筛选）和 `predict_tournament`（分组淘汰，不需要嵌入模型）。
- **吞吐优化**：重复负载可开启 `compile=True`、`pad_to_multiple=16`、`cache_prompts=True`。
- **工具与服务**：命令行（`laya-mlx predict`）、HTTP（`laya-mlx-serve`）、MCP（`laya-mlx-mcp-server`）、评估与校准，以及 LangChain/LlamaIndex/CrewAI/Agno 集成，按需安装 extras。
- **导出**：`laya-mlx convert` 生成独立的 MLX 模型目录。

## 性能

M3 Max（40 核 GPU，128 GiB），FP16，端到端：

| | Laya 421M | 多语言 322M |
|---|---:|---:|
| 单个短问题 P50 / P95 | **13.42 / 13.92 ms** | **7.39 / 7.79 ms** |
| 50 个问题吞吐 | **146.8 问/秒** | **395.0 问/秒** |
| 单个问题 MLX 峰值内存 | 943.6 MiB | 687.6 MiB |

计时包含提示词构造、分词、推理、校准和结果格式化，不含模型加载。吞吐一行使用
`batch_size=64`（默认 16）。实际延迟随输入长度、问题数量和系统负载变化。

三个模型在 FP32 和 FP16 下都与上游在 **63/63 道验证题**上选出相同答案（共 378/378 次比对），
重复调用结果一致，未测到内存增长。这衡量的是移植是否忠实，不是任务准确率。
[测量方法、全部计时样本与概率误差](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md)。

## 贪吃蛇演示

上面的 GIF 是一次真实的本地运行，原速录制：每一步都向 Laya 提三个问题，
画面上可见的安全层会拦下危险的走法。

```bash
pip install 'laya-mlx[demo]'
hf download aac6fef/laya-multilingual-mlx
laya-snake                      # 按节奏运行；加 --max-speed 取消节奏限制
laya-snake --optimize --max-speed
```

终端至少 104 × 35。空格暂停，↑/↓ 调速，R 重开，Q 退出。开启 `--optimize` 后，
在 M3 Max 上实测 **2,400 步内达到 75.4 步/秒**，零死亡。
[演示说明](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) ·
[速度与稳定性](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_BENCHMARKS.md) ·
[优化细节](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_OPTIMIZATION.md)。

## 与上游的关系

这是独立移植，不是 Convai Innovations 的官方发布。提示词格式、校准、路由、输出结构和主机侧 API
跟随 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) **v0.5.4**；编码器和决策头用 MLX
重新实现。CUDA、ONNX、PyTorch 后端控制、vegaml 引擎和训练不在范围内，RLCD 训练与微调请使用上游项目。
[原生边界](https://github.com/mizorewww/laya-mlx/blob/main/docs/USAGE.md#upstream-compatibility) ·
[上游兼容性记录](https://github.com/mizorewww/laya-mlx/blob/main/docs/UPSTREAM.md) ·
[全部文档](https://github.com/mizorewww/laya-mlx/blob/main/docs/README.md)。

## 参与贡献

欢迎提 issue 和 PR。与上游保持一致的修复优先于新功能。开发环境、测试、基准和发版流程见
[CONTRIBUTING.md](https://github.com/mizorewww/laya-mlx/blob/main/CONTRIBUTING.md)。

## 许可证

Apache-2.0，见 [LICENSE](https://github.com/mizorewww/laya-mlx/blob/main/LICENSE) 与
[NOTICE](https://github.com/mizorewww/laya-mlx/blob/main/NOTICE)。Laya 及其预训练权重归 Convai Innovations
和上游贡献者所有。神经网络结构参照 Laya 与 Hugging Face ModernBERT 实现。
