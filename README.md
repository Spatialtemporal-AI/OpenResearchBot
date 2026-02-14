<div align="center">
  <h1>🔬 OpenResearchBot: VLA 研究助手</h1>
  <p>
    <strong>基于 <a href="https://github.com/HKUDS/nanobot">nanobot</a> 框架开发的 VLA (Vision-Language-Action) 研究追踪助手</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/based%20on-nanobot-orange" alt="Based on nanobot">
  </p>
</div>

---

## 📖 项目简介

OpenResearchBot 是在 [nanobot](https://github.com/HKUDS/nanobot) 超轻量 AI Agent 框架基础上扩展的**科研实验追踪助手**，专为 VLA（Vision-Language-Action）模型研究场景设计，同时也适用于一般的机器学习/深度学习实验管理。

### 🎯 核心目标

- 帮助研究人员通过 AI Agent **自然语言对话**来管理科研任务和实验进度
- 提供结构化的训练运行记录，特别支持 VLA 模型特有的字段（动作空间、观察空间、具身化平台等）
- 可视化实验数据，支持终端纯文本图表和交互式 HTML 仪表盘两种模式

---

## ✨ 新增功能概览

本项目在原版 nanobot 基础上新增了以下功能模块：

| 模块 | 文件 | 说明 |
|------|------|------|
| 🧪 训练追踪器 | `nanobot/agent/tools/training_tracker.py` | 训练运行全生命周期管理，支持 VLA 专属字段 |
| 📋 任务追踪器 | `nanobot/agent/tools/task_tracker.py` | 科研任务管理（todo/doing/done/blocked） |
| 📊 纯文本可视化 | `nanobot/agent/tools/text_viz.py` | 终端/聊天中渲染柱状图、折线图、Sparkline |
| 🌐 HTML 仪表盘 | `nanobot/agent/tools/html_dashboard.py` | 基于 Chart.js 的交互式可视化仪表盘 |
| 🖥️ CLI 工具 | `nanobot/cli_tracker.py` | 独立命令行入口，无需启动 Agent 即可查看数据 |

### 修改的原有文件

| 文件 | 修改内容 |
|------|---------|
| `nanobot/__init__.py` | 增加 Windows 终端 emoji 兼容处理 |
| `nanobot/agent/loop.py` | 注册 TaskTrackerTool 和 TrainingTrackerTool |
| `workspace/AGENTS.md` | 更新 Agent 身份为 VLA 研究助手，添加工具使用指南 |
| `workspace/SOUL.md` | 更新 Agent 人格为科研导向 |

---

## 🧪 功能一：训练运行追踪器（Training Tracker）

**文件**：`nanobot/agent/tools/training_tracker.py`（685 行）

专门为 VLA 模型训练设计的追踪工具，同时也支持任意 ML/DL 训练。

### 支持的操作

| 操作 | 说明 |
|------|------|
| `create` | 创建新的训练运行，记录模型、数据集、超参数、VLA 配置 |
| `update` | 更新训练状态（queued/running/completed/failed/stopped） |
| `log_metrics` | 记录训练指标（loss, success_rate, 或自定义指标） |
| `list` | 按状态/模型筛选查看训练列表 |
| `detail` | 查看单次训练的完整信息（含 VLA 配置和指标历史） |
| `compare` | 多次训练横向对比 |
| `delete` | 删除训练记录 |
| `summary` | 总览统计，含最佳表现运行 |
| `visualize` | 纯文本可视化训练曲线 |
| `dashboard` | 生成交互式 HTML 仪表盘 |

### VLA 专属字段

通过 `vla_config` 参数记录 VLA 模型特有信息：

```json
{
  "action_space": "7-DoF delta EEF",
  "observation_space": "RGB 256x256 + proprioception",
  "embodiment": "Franka Panda",
  "environment": "real-world tabletop",
  "task_suite": "pick-and-place, drawer open/close",
  "action_tokenizer": "256 bins per dim",
  "backbone": "Llama-2-7B"
}
```

### 使用示例（Agent 对话）

```
用户：帮我记录一下 OpenVLA-7B 在 Bridge 数据集上的训练，学习率 2e-5，batch size 16

Agent（自动调用 training_tracker）：
  ✅ 已创建训练运行 [run-a1b2c3]
  模型：OpenVLA-7B | 数据集：bridge_v2 | 状态：running

用户：loss 降到 0.35 了，success rate 72%

Agent（自动调用 log_metrics）：
  📊 已记录指标 → run-a1b2c3
  loss: 0.35 | success_rate: 72.0%
```

---

## 📋 功能二：任务追踪器（Task Tracker）

**文件**：`nanobot/agent/tools/task_tracker.py`（355 行）

管理科研任务和未完成工作的结构化工具。

### 支持的操作

| 操作 | 说明 |
|------|------|
| `create` | 创建任务，设置标题、描述、优先级、标签 |
| `update` | 更新状态（todo/doing/done/blocked），添加时间戳备注 |
| `list` | 按状态/标签筛选任务列表 |
| `detail` | 查看任务完整详情 |
| `delete` | 删除任务 |
| `summary` | 任务总览统计 |
| `visualize` | 纯文本可视化任务面板 |
| `dashboard` | 生成交互式 HTML 仪表盘 |

### 任务状态流转

```
todo → doing → done
  ↓       ↓
blocked  blocked
```

### 使用示例（Agent 对话）

```
用户：还没复现 OpenVLA 论文 Table 2 的实验，帮我记一下

Agent（自动调用 task_tracker）：
  ✅ 已创建任务 [t-x1y2z3]
  标题：复现 OpenVLA 论文 Table 2 实验
  优先级：high | 状态：todo | 标签：VLA, 复现

用户：开始做了

Agent（自动调用 task_tracker update）：
  📋 任务已更新 → doing
```

---

## 📊 功能三：纯文本可视化（Text Visualization）

**文件**：`nanobot/agent/tools/text_viz.py`（672 行）

在终端或聊天界面中渲染各种数据图表，**无需任何外部依赖**。

### 支持的图表类型

| 图表 | 函数 | 说明 |
|------|------|------|
| 📊 柱状图 | `bar_chart()` | 水平柱状图，展示分类数据 |
| 📈 折线图 | `line_chart()` | ASCII 折线图，支持多系列 |
| ⚡ Sparkline | `sparkline()` | 紧凑单行趋势线 |
| 🏅 排行榜 | `leaderboard()` | 训练运行排名 |

### 渲染效果示例

```
📊 Task Status
──────────────────────────────────────
todo     ████████░░░░░  3
doing    ████░░░░░░░░░  2
done     █████████████  5
blocked  ██░░░░░░░░░░░  1

📈 Training Loss
loss  ▇▆▅▄▃▂▂▁  0.19
```

---

## 🌐 功能四：HTML 交互式仪表盘（HTML Dashboard）

**文件**：`nanobot/agent/tools/html_dashboard.py`（794 行）

生成自包含的 HTML 仪表盘文件，基于 Chart.js CDN，无需搭建服务器，浏览器直接打开即可。

### 特性

- 📱 响应式设计，适配桌面和移动端
- 🎨 深色主题，现代化 UI
- 📊 交互式图表（Chart.js 4.x）
- 🔄 任务状态分布饼图
- 📈 训练指标折线图
- 🏆 训练运行对比表格
- 🌐 纯静态 HTML，可离线查看

### 仪表盘页面

- **完整仪表盘**：任务 + 训练一体化视图
- **任务仪表盘**：仅展示任务状态和进度
- **训练仪表盘**：训练指标可视化和运行对比

---

## 🖥️ 功能五：独立 CLI 工具（CLI Tracker）

**文件**：`nanobot/cli_tracker.py`（170 行）

无需启动 Agent 即可在命令行中查看和操作追踪数据。

### 使用方法

```bash
# 📊 打开完整 HTML 仪表盘（自动打开浏览器）
python -m nanobot.cli_tracker dashboard

# 📋 任务相关
python -m nanobot.cli_tracker task visualize        # 文本模式可视化
python -m nanobot.cli_tracker task list              # 查看任务列表
python -m nanobot.cli_tracker task summary           # 查看任务总结
python -m nanobot.cli_tracker task dashboard         # 打开任务 HTML 仪表盘

# 🧪 训练相关
python -m nanobot.cli_tracker train visualize        # 文本模式可视化
python -m nanobot.cli_tracker train summary          # 查看训练总结
python -m nanobot.cli_tracker train dashboard        # 打开训练 HTML 仪表盘
python -m nanobot.cli_tracker train visualize --run-id run-abc123       # 查看指定训练
python -m nanobot.cli_tracker train visualize --run-ids run-abc run-def # 对比多个训练
```

---

## 🔧 其他改进

### Windows 兼容性

修改了 `nanobot/__init__.py`，增加 Windows 终端 emoji 兼容处理，在不支持 emoji 的终端上自动降级为文本标识。

### Agent 身份定制

- 将 Agent 身份从通用助手更新为 VLA 研究助手
- Agent 会主动建议追踪任务和训练运行
- 熟悉 VLA 研究术语（动作空间、具身化、sim-to-real 等）

---

## 📁 新增文件结构

```
nanobot/
├── agent/tools/
│   ├── training_tracker.py   # 🧪 训练运行追踪器（685 行）
│   ├── task_tracker.py       # 📋 任务追踪器（355 行）
│   ├── text_viz.py           # 📊 纯文本可视化（672 行）
│   └── html_dashboard.py     # 🌐 HTML 仪表盘生成器（794 行）
├── cli_tracker.py            # 🖥️ 独立 CLI 工具（170 行）
workspace/
├── AGENTS.md                 # 更新：研究助手指令
├── SOUL.md                   # 更新：VLA 研究人格
└── research/                 # 数据存储目录
    ├── tasks.json            # 任务数据
    ├── training_runs.json    # 训练运行数据
    └── dashboard.html        # 生成的仪表盘
tests/
└── test_trackers.py          # 追踪器功能测试
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 配置 API Key

参考 [nanobot 文档](https://github.com/HKUDS/nanobot) 配置 LLM Provider。

### 3. 通过 Agent 对话使用

```bash
nanobot agent
```

在对话中自然地提及实验和任务，Agent 会自动调用追踪工具：

```
> 帮我创建一个训练任务：微调 OpenVLA-7B，数据集 bridge_v2，学习率 2e-5
> 记录一下当前 loss 0.45，success rate 65%
> 对比一下最近的两次训练
> 还有哪些任务没完成？
```

### 4. 通过 CLI 直接查看

```bash
python -m nanobot.cli_tracker dashboard    # 打开 HTML 仪表盘
python -m nanobot.cli_tracker task list    # 查看任务
python -m nanobot.cli_tracker train summary # 训练总结
```

---

## 🏗️ 技术实现

- **数据存储**：JSON 文件存储在 `workspace/research/` 目录下，轻量且可版本控制
- **工具注册**：通过 nanobot 的 Tool 基类实现，自动集成到 Agent 的工具链中
- **可视化**：
  - 纯文本模式使用 Unicode 字符渲染，零依赖
  - HTML 模式使用 Chart.js CDN，生成自包含 HTML 文件
- **VLA 支持**：通过 `vla_config` 字段扩展，不影响通用训练追踪功能

---

## 📜 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

- [nanobot](https://github.com/HKUDS/nanobot) — 底层 AI Agent 框架
- [Chart.js](https://www.chartjs.org/) — HTML 仪表盘图表库
