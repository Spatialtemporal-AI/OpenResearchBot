<div align="center">
  <h1>🔬 OpenResearchBot: VLA 研究助手</h1>
  <p>
    <strong>基于 <a href="https://github.com/HKUDS/nanobot">nanobot</a> 框架开发的 VLA (Vision-Language-Action) 研究追踪助手</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/based%20on-nanobot-orange" alt="Based on nanobot">
    <img src="https://img.shields.io/badge/飞书-Feishu%20Bot-4e6ef2" alt="Feishu Bot">
  </p>
</div>

---

## 📖 项目简介

OpenResearchBot 是在 [nanobot](https://github.com/HKUDS/nanobot) 超轻量 AI Agent 框架基础上扩展的**科研实验追踪助手**，专为 VLA（Vision-Language-Action）模型研究场景设计，同时也适用于一般的机器学习/深度学习实验管理。

- 通过 AI Agent **自然语言对话**管理科研任务和实验进度
- 结构化训练运行记录，特别支持 VLA 模型特有字段（动作空间、观察空间、具身化平台等）
- 可视化实验数据：终端纯文本图表 + 交互式 HTML 仪表盘
- **飞书机器人**：随时随地通过飞书与 Agent 对话，手机端也能管理实验

---

## ✨ 功能概览

| 模块 | 文件 | 说明 |
|------|------|------|
| 🧪 训练追踪器 | `nanobot/agent/tools/training_tracker.py` | 训练运行全生命周期管理，支持 VLA 专属字段 |
| 📋 任务追踪器 | `nanobot/agent/tools/task_tracker.py` | 科研任务管理（todo/doing/done/blocked） |
| 📊 纯文本可视化 | `nanobot/agent/tools/text_viz.py` | 终端/聊天中渲染柱状图、折线图、Sparkline |
| 🌐 HTML 仪表盘 | `nanobot/agent/tools/html_dashboard.py` | 基于 Chart.js 的交互式可视化仪表盘 |
| 🖥️ CLI 工具 | `nanobot/cli_tracker.py` | 独立命令行入口，含实时仪表盘服务器 |
| 🔴 自动训练记录 | `nanobot/tracker.py` | 训练脚本加几行代码即可自动记录，支持 PyTorch / HuggingFace / Lightning |
| 💬 飞书机器人 | `nanobot/channels/feishu.py` | 飞书/Lark 频道，WebSocket 长连接，卡片消息 |
| 🚀 飞书启动器 | `nanobot/feishu_bot.py` | 独立飞书 Bot 入口，含实时仪表盘服务 |

---

## 🧪 训练运行追踪器

专为 VLA 模型训练设计，同时支持任意 ML/DL 训练。支持操作：`create` / `update` / `log_metrics` / `list` / `detail` / `compare` / `summary` / `visualize` / `dashboard`。

通过 `vla_config` 记录 VLA 特有信息（action_space、observation_space、embodiment、environment 等）。

```
用户：帮我记录一下 OpenVLA-7B 在 Bridge 数据集上的训练，学习率 2e-5，batch size 16
Agent：✅ 已创建训练运行 [run-a1b2c3] 模型：OpenVLA-7B | 数据集：bridge_v2

用户：loss 降到 0.35 了，success rate 72%
Agent：📊 已记录指标 → run-a1b2c3  loss: 0.35 | success_rate: 72.0%
```

## 📋 任务追踪器

管理科研任务（`todo → doing → done / blocked`），支持优先级、标签、时间戳备注。

## 📊 可视化

- **纯文本模式**：终端直接渲染柱状图、折线图、Sparkline，零依赖
- **HTML 仪表盘**：Chart.js 交互式图表，深色主题，响应式设计，浏览器直接打开
- **实时仪表盘**：每 3 秒自动刷新，训练过程中保持打开即可实时监控

## 💬 飞书机器人

通过飞书与 Agent 直接对话。基于 **WebSocket 长连接**，**无需公网 IP**，开箱即用。

| 特性 | 说明 |
|------|------|
| 🔌 WebSocket 长连接 | 无需公网 IP、无需 Webhook |
| 🃏 交互式卡片消息 | Markdown + 原生表格渲染 |
| ⏳ "思考中" 指示器 | 处理时显示，完成后原地更新为回复 |
| ⚡ 快捷命令 | `/help` `/tasks` `/trains` `/dashboard` `/status` |
| 📊 实时仪表盘 | 启动时自动开启 HTTP 仪表盘服务，LAN 内手机可访问 |

## 🔴 自动训练记录

在训练脚本中加几行代码，即可自动记录训练全过程。**无需启动 Agent**，数据直接写入 JSON，Dashboard 和 Agent 都能实时看到。

### PyTorch 原生训练循环

```python
from nanobot.tracker import NanobotTracker

# 方式一：with 语句（推荐，异常时自动标记 failed，正常退出标记 completed）
with NanobotTracker(
    name="OpenVLA-7B finetune Bridge",
    model="OpenVLA-7B",
    dataset="bridge_v2",
    hyperparams={"lr": 2e-5, "batch_size": 32, "epochs": 100},
    # gpu_info 自动检测，vla_config 可选
) as tracker:
    for epoch in range(100):
        loss = train_one_epoch()
        tracker.log(epoch=epoch, loss=loss)                          # 记录指标
        tracker.log(epoch=epoch, eval_loss=val_loss, success_rate=sr) # 可多次调用
        tracker.log_checkpoint(f"ckpt/epoch_{epoch}.pt")             # 记录 checkpoint

# 方式二：手动管理
tracker = NanobotTracker(name="my-exp", model="OpenVLA-7B")
for epoch in range(100):
    tracker.log(epoch=epoch, loss=loss)
tracker.finish()  # 或 tracker.fail() / tracker.stop()
```

### HuggingFace Trainer 集成

```python
from nanobot.tracker import NanobotHFCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[NanobotHFCallback(name="my-experiment", model="OpenVLA-7B")],
)
trainer.train()  # 自动记录所有 loss、eval metrics、checkpoint
```

### PyTorch Lightning 集成

```python
from nanobot.tracker import NanobotLightningCallback

trainer = pl.Trainer(
    callbacks=[NanobotLightningCallback(name="my-exp", model="OpenVLA-7B")],
)
trainer.fit(model)  # 自动记录每个 epoch 的指标
```

### 功能特性

| 特性 | 说明 |
|------|------|
| 🔍 自动检测 GPU | 自动获取 GPU 型号和显存信息 |
| 🛡️ 异常安全 | with 语句或 atexit 兜底，进程崩溃也能记录状态 |
| 📝 灵活日志 | 任意 key-value 指标，不限制字段名 |
| ⚡ 写入频率可控 | `log_every_n_steps` 控制磁盘写入频率 |
| 🔄 与 Agent 互通 | 数据和手动创建的记录在同一文件，Agent 可查询/对比 |
| 🌐 Dashboard 实时可见 | 启动 live dashboard 后自动刷新显示 |

---

## 📁 文件结构

```
nanobot/
├── tracker.py               # 🔴 自动训练记录（PyTorch/HF/Lightning）
├── feishu_bot.py             # 飞书 Bot 独立入口
├── cli_tracker.py            # CLI 工具（含 live 实时服务器）
├── agent/tools/
│   ├── training_tracker.py   # 训练运行追踪器
│   ├── task_tracker.py       # 任务追踪器
│   ├── text_viz.py           # 纯文本可视化
│   └── html_dashboard.py     # HTML 仪表盘生成器
├── channels/
│   └── feishu.py             # 飞书频道（WebSocket + 卡片消息）
workspace/
├── AGENTS.md                 # Agent 指令
├── SOUL.md                   # Agent 人格
└── research/                 # 数据存储
    ├── tasks.json
    ├── training_runs.json
    └── dashboard.html
```

---

## 🚀 快速开始

### 1. 安装

```bash
pip install -e .
pip install lark-oapi>=1.0.0   # 飞书机器人需要
```

### 2. 配置

参考 [nanobot 文档](https://github.com/HKUDS/nanobot) 配置 LLM Provider（`~/.nanobot/config.json`）。

### 3. 使用方式

```bash
# 方式一：终端 Agent 对话
nanobot agent

# 方式二：飞书机器人（推荐 📱 随时随地使用）
python -m nanobot.feishu_bot

# 方式三：CLI 工具
python -m nanobot.cli_tracker live         # 实时仪表盘
python -m nanobot.cli_tracker task list    # 查看任务
python -m nanobot.cli_tracker train summary # 训练总结
```

---

## 📜 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

- [nanobot](https://github.com/HKUDS/nanobot) — 底层 AI Agent 框架
- [Chart.js](https://www.chartjs.org/) — HTML 仪表盘图表库
- [lark-oapi](https://github.com/larksuite/oapi-sdk-python) — 飞书/Lark 开放平台 Python SDK
