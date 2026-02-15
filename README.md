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

<<<<<<< HEAD
---
=======
🐈 **nanobot** is an **ultra-lightweight** personal AI assistant inspired by [OpenClaw](https://github.com/openclaw/openclaw) 
>>>>>>> origin/feature/video_analysis

## 📖 项目简介

<<<<<<< HEAD
OpenResearchBot 是在 [nanobot](https://github.com/HKUDS/nanobot) 超轻量 AI Agent 框架基础上扩展的**科研实验追踪助手**，专为 VLA（Vision-Language-Action）模型研究场景设计，同时也适用于一般的机器学习/深度学习实验管理。
=======
📏 Real-time line count: **3,510 lines** (run `bash core_agent_lines.sh` to verify anytime)
>>>>>>> origin/feature/video_analysis

- 通过 AI Agent **自然语言对话**管理科研任务和实验进度
- 结构化训练运行记录，特别支持 VLA 模型特有字段（动作空间、观察空间、具身化平台等）
- 可视化实验数据：终端纯文本图表 + 交互式 HTML 仪表盘
- **飞书机器人**：随时随地通过飞书与 Agent 对话，手机端也能管理实验

<<<<<<< HEAD
---
=======
- **2026-02-10** 🎉 Released v0.1.3.post6 with improvements! Check the updates [notes](https://github.com/HKUDS/nanobot/releases/tag/v0.1.3.post6) and our [roadmap](https://github.com/HKUDS/nanobot/discussions/431).
- **2026-02-09** 💬 Added Slack, Email, and QQ support — nanobot now supports multiple chat platforms!
- **2026-02-08** 🔧 Refactored Providers—adding a new LLM provider now takes just 2 simple steps! Check [here](#providers).
- **2026-02-07** 🚀 Released v0.1.3.post5 with Qwen support & several key improvements! Check [here](https://github.com/HKUDS/nanobot/releases/tag/v0.1.3.post5) for details.
- **2026-02-06** ✨ Added Moonshot/Kimi provider, Discord integration, and enhanced security hardening!
- **2026-02-05** ✨ Added Feishu channel, DeepSeek provider, and enhanced scheduled tasks support!
- **2026-02-04** 🚀 Released v0.1.3.post4 with multi-provider & Docker support! Check [here](https://github.com/HKUDS/nanobot/releases/tag/v0.1.3.post4) for details.
- **2026-02-03** ⚡ Integrated vLLM for local LLM support and improved natural language task scheduling!
- **2026-02-02** 🎉 nanobot officially launched! Welcome to try 🐈 nanobot!
>>>>>>> origin/feature/video_analysis

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

### (Optional) Install Cursor CLI integration

If you want the agent to optionally use **Cursor CLI Agent** (in Ask/Plan modes)
to help figure out how to run code in a given folder or which changes are needed
before running, install Cursor CLI (`agent`) as well:

```bash
# macOS, Linux, and Windows (WSL)
curl https://cursor.com/install -fsS | bash

# then make sure ~/.local/bin is on your PATH, e.g. for bash:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# verify installation
agent --version
```

After installation, the nanobot agent can call Cursor CLI via the
`cursor_cli_ask` tool. Use it when you are unsure which command to run
for a specific project directory, or when code may need modifications
before it can run. The tool runs `agent` in non-interactive Ask/Plan mode
inside the target directory and returns its analysis back to the main agent.

## 🚀 Quick Start

<<<<<<< HEAD
### 💬 飞书机器人
=======
> [!TIP]
> Set your API key in `~/.nanobot/config.json`.
> Get API keys: [OpenRouter](https://openrouter.ai/keys) (Global) · [Brave Search](https://brave.com/search/api/) (optional, for web search)
>>>>>>> origin/feature/video_analysis

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

<<<<<<< HEAD
### PyTorch 原生训练循环
=======
**3. Chat**

```bash
nanobot agent -m "What is 2+2?"
```

That's it! You have a working AI assistant in 2 minutes.

## 🖥️ Local Models (vLLM)

Run nanobot with your own local models using vLLM or any OpenAI-compatible server.

**1. Start your vLLM server**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**2. Configure** (`~/.nanobot/config.json`)

```json
{
  "providers": {
    "vllm": {
      "apiKey": "dummy",
      "apiBase": "http://localhost:8000/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "meta-llama/Llama-3.1-8B-Instruct"
    }
  }
}
```

**3. Chat**

```bash
nanobot agent -m "Hello from my local LLM!"
```

> [!TIP]
> The `apiKey` can be any non-empty string for local servers that don't require authentication.

## 💬 Chat Apps

Talk to your nanobot through Telegram, Discord, WhatsApp, Feishu, Mochat, DingTalk, Slack, Email, or QQ — anytime, anywhere.

| Channel | Setup |
|---------|-------|
| **Telegram** | Easy (just a token) |
| **Discord** | Easy (bot token + intents) |
| **WhatsApp** | Medium (scan QR) |
| **Feishu** | Medium (app credentials) |
| **Mochat** | Medium (claw token + websocket) |
| **DingTalk** | Medium (app credentials) |
| **Slack** | Medium (bot + app tokens) |
| **Email** | Medium (IMAP/SMTP credentials) |
| **QQ** | Easy (app credentials) |

<details>
<summary><b>Telegram</b> (Recommended)</summary>

**1. Create a bot**
- Open Telegram, search `@BotFather`
- Send `/newbot`, follow prompts
- Copy the token

**2. Configure**

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

> You can find your **User ID** in Telegram settings. It is shown as `@yourUserId`.
> Copy this value **without the `@` symbol** and paste it into the config file.


**3. Run**

```bash
nanobot gateway
```

</details>

<details>
<summary><b>Mochat (Claw IM)</b></summary>

Uses **Socket.IO WebSocket** by default, with HTTP polling fallback.

**1. Ask nanobot to set up Mochat for you**

Simply send this message to nanobot (replace `xxx@xxx` with your real email):

```
Read https://raw.githubusercontent.com/HKUDS/MoChat/refs/heads/main/skills/nanobot/skill.md and register on MoChat. My Email account is xxx@xxx Bind me as your owner and DM me on MoChat.
```

nanobot will automatically register, configure `~/.nanobot/config.json`, and connect to Mochat.

**2. Restart gateway**

```bash
nanobot gateway
```

That's it — nanobot handles the rest!

<br>

<details>
<summary>Manual configuration (advanced)</summary>

If you prefer to configure manually, add the following to `~/.nanobot/config.json`:

> Keep `claw_token` private. It should only be sent in `X-Claw-Token` header to your Mochat API endpoint.

```json
{
  "channels": {
    "mochat": {
      "enabled": true,
      "base_url": "https://mochat.io",
      "socket_url": "https://mochat.io",
      "socket_path": "/socket.io",
      "claw_token": "claw_xxx",
      "agent_user_id": "6982abcdef",
      "sessions": ["*"],
      "panels": ["*"],
      "reply_delay_mode": "non-mention",
      "reply_delay_ms": 120000
    }
  }
}
```



</details>

</details>

<details>
<summary><b>Discord</b></summary>

**1. Create a bot**
- Go to https://discord.com/developers/applications
- Create an application → Bot → Add Bot
- Copy the bot token

**2. Enable intents**
- In the Bot settings, enable **MESSAGE CONTENT INTENT**
- (Optional) Enable **SERVER MEMBERS INTENT** if you plan to use allow lists based on member data

**3. Get your User ID**
- Discord Settings → Advanced → enable **Developer Mode**
- Right-click your avatar → **Copy User ID**

**4. Configure**

```json
{
  "channels": {
    "discord": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

**5. Invite the bot**
- OAuth2 → URL Generator
- Scopes: `bot`
- Bot Permissions: `Send Messages`, `Read Message History`
- Open the generated invite URL and add the bot to your server

**6. Run**

```bash
nanobot gateway
```

</details>

<details>
<summary><b>WhatsApp</b></summary>

Requires **Node.js ≥18**.

**1. Link device**

```bash
nanobot channels login
# Scan QR with WhatsApp → Settings → Linked Devices
```

**2. Configure**

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

**3. Run** (two terminals)

```bash
# Terminal 1
nanobot channels login

# Terminal 2
nanobot gateway
```

</details>

<details>
<summary><b>Feishu (飞书)</b></summary>

Uses **WebSocket** long connection — no public IP required.

**1. Create a Feishu bot**
- Visit [Feishu Open Platform](https://open.feishu.cn/app)
- Create a new app → Enable **Bot** capability
- **Permissions**: Add `im:message` (send messages)
- **Events**: Add `im.message.receive_v1` (receive messages)
  - Select **Long Connection** mode (requires running nanobot first to establish connection)
- Get **App ID** and **App Secret** from "Credentials & Basic Info"
- Publish the app

**2. Configure**

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "encryptKey": "",
      "verificationToken": "",
      "allowFrom": []
    }
  }
}
```

> `encryptKey` and `verificationToken` are optional for Long Connection mode.
> `allowFrom`: Leave empty to allow all users, or add `["ou_xxx"]` to restrict access.

**3. Run**

```bash
nanobot gateway
```

> [!TIP]
> Feishu uses WebSocket to receive messages — no webhook or public IP needed!

</details>

<details>
<summary><b>QQ (QQ单聊)</b></summary>

Uses **botpy SDK** with WebSocket — no public IP required. Currently supports **private messages only**.

**1. Register & create bot**
- Visit [QQ Open Platform](https://q.qq.com) → Register as a developer (personal or enterprise)
- Create a new bot application
- Go to **开发设置 (Developer Settings)** → copy **AppID** and **AppSecret**

**2. Set up sandbox for testing**
- In the bot management console, find **沙箱配置 (Sandbox Config)**
- Under **在消息列表配置**, click **添加成员** and add your own QQ number
- Once added, scan the bot's QR code with mobile QQ → open the bot profile → tap "发消息" to start chatting

**3. Configure**

> - `allowFrom`: Leave empty for public access, or add user openids to restrict. You can find openids in the nanobot logs when a user messages the bot.
> - For production: submit a review in the bot console and publish. See [QQ Bot Docs](https://bot.q.qq.com/wiki/) for the full publishing flow.

```json
{
  "channels": {
    "qq": {
      "enabled": true,
      "appId": "YOUR_APP_ID",
      "secret": "YOUR_APP_SECRET",
      "allowFrom": []
    }
  }
}
```

**4. Run**

```bash
nanobot gateway
```

Now send a message to the bot from QQ — it should respond!

</details>

<details>
<summary><b>DingTalk (钉钉)</b></summary>

Uses **Stream Mode** — no public IP required.

**1. Create a DingTalk bot**
- Visit [DingTalk Open Platform](https://open-dev.dingtalk.com/)
- Create a new app -> Add **Robot** capability
- **Configuration**:
  - Toggle **Stream Mode** ON
- **Permissions**: Add necessary permissions for sending messages
- Get **AppKey** (Client ID) and **AppSecret** (Client Secret) from "Credentials"
- Publish the app

**2. Configure**

```json
{
  "channels": {
    "dingtalk": {
      "enabled": true,
      "clientId": "YOUR_APP_KEY",
      "clientSecret": "YOUR_APP_SECRET",
      "allowFrom": []
    }
  }
}
```

> `allowFrom`: Leave empty to allow all users, or add `["staffId"]` to restrict access.

**3. Run**

```bash
nanobot gateway
```

</details>

<details>
<summary><b>Slack</b></summary>

Uses **Socket Mode** — no public URL required.

**1. Create a Slack app**
- Go to [Slack API](https://api.slack.com/apps) → **Create New App** → "From scratch"
- Pick a name and select your workspace

**2. Configure the app**
- **Socket Mode**: Toggle ON → Generate an **App-Level Token** with `connections:write` scope → copy it (`xapp-...`)
- **OAuth & Permissions**: Add bot scopes: `chat:write`, `reactions:write`, `app_mentions:read`
- **Event Subscriptions**: Toggle ON → Subscribe to bot events: `message.im`, `message.channels`, `app_mention` → Save Changes
- **App Home**: Scroll to **Show Tabs** → Enable **Messages Tab** → Check **"Allow users to send Slash commands and messages from the messages tab"**
- **Install App**: Click **Install to Workspace** → Authorize → copy the **Bot Token** (`xoxb-...`)

**3. Configure nanobot**

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "botToken": "xoxb-...",
      "appToken": "xapp-...",
      "groupPolicy": "mention"
    }
  }
}
```

**4. Run**

```bash
nanobot gateway
```

DM the bot directly or @mention it in a channel — it should respond!

> [!TIP]
> - `groupPolicy`: `"mention"` (default — respond only when @mentioned), `"open"` (respond to all channel messages), or `"allowlist"` (restrict to specific channels).
> - DM policy defaults to open. Set `"dm": {"enabled": false}` to disable DMs.

</details>

<details>
<summary><b>Email</b></summary>

Give nanobot its own email account. It polls **IMAP** for incoming mail and replies via **SMTP** — like a personal email assistant.

**1. Get credentials (Gmail example)**
- Create a dedicated Gmail account for your bot (e.g. `my-nanobot@gmail.com`)
- Enable 2-Step Verification → Create an [App Password](https://myaccount.google.com/apppasswords)
- Use this app password for both IMAP and SMTP

**2. Configure**

> - `consentGranted` must be `true` to allow mailbox access. This is a safety gate — set `false` to fully disable.
> - `allowFrom`: Leave empty to accept emails from anyone, or restrict to specific senders.
> - `smtpUseTls` and `smtpUseSsl` default to `true` / `false` respectively, which is correct for Gmail (port 587 + STARTTLS). No need to set them explicitly.
> - Set `"autoReplyEnabled": false` if you only want to read/analyze emails without sending automatic replies.

```json
{
  "channels": {
    "email": {
      "enabled": true,
      "consentGranted": true,
      "imapHost": "imap.gmail.com",
      "imapPort": 993,
      "imapUsername": "my-nanobot@gmail.com",
      "imapPassword": "your-app-password",
      "smtpHost": "smtp.gmail.com",
      "smtpPort": 587,
      "smtpUsername": "my-nanobot@gmail.com",
      "smtpPassword": "your-app-password",
      "fromAddress": "my-nanobot@gmail.com",
      "allowFrom": ["your-real-email@gmail.com"]
    }
  }
}
```


**3. Run**

```bash
nanobot gateway
```

</details>

## ⚙️ Configuration

Config file: `~/.nanobot/config.json`

### Providers

> [!TIP]
> - **Groq** provides free voice transcription via Whisper. If configured, Telegram voice messages will be automatically transcribed.
> - **Zhipu Coding Plan**: If you're on Zhipu's coding plan, set `"apiBase": "https://open.bigmodel.cn/api/coding/paas/v4"` in your zhipu provider config.
> - **MiniMax (Mainland China)**: If your API key is from MiniMax's mainland China platform (minimaxi.com), set `"apiBase": "https://api.minimaxi.com/v1"` in your minimax provider config.

| Provider | Purpose | Get API Key |
|----------|---------|-------------|
| `openrouter` | LLM (recommended, access to all models) | [openrouter.ai](https://openrouter.ai) |
| `anthropic` | LLM (Claude direct) | [console.anthropic.com](https://console.anthropic.com) |
| `openai` | LLM (GPT direct) | [platform.openai.com](https://platform.openai.com) |
| `deepseek` | LLM (DeepSeek direct) | [platform.deepseek.com](https://platform.deepseek.com) |
| `groq` | LLM + **Voice transcription** (Whisper) | [console.groq.com](https://console.groq.com) |
| `gemini` | LLM (Gemini direct) | [aistudio.google.com](https://aistudio.google.com) |
| `minimax` | LLM (MiniMax direct) | [platform.minimax.io](https://platform.minimax.io) |
| `aihubmix` | LLM (API gateway, access to all models) | [aihubmix.com](https://aihubmix.com) |
| `dashscope` | LLM (Qwen) | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) |
| `moonshot` | LLM (Moonshot/Kimi) | [platform.moonshot.cn](https://platform.moonshot.cn) |
| `zhipu` | LLM (Zhipu GLM) | [open.bigmodel.cn](https://open.bigmodel.cn) |
| `vllm` | LLM (local, any OpenAI-compatible server) | — |

<details>
<summary><b>Adding a New Provider (Developer Guide)</b></summary>

nanobot uses a **Provider Registry** (`nanobot/providers/registry.py`) as the single source of truth.
Adding a new provider only takes **2 steps** — no if-elif chains to touch.

**Step 1.** Add a `ProviderSpec` entry to `PROVIDERS` in `nanobot/providers/registry.py`:
>>>>>>> origin/feature/video_analysis

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

<<<<<<< HEAD
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
=======
</details>


### Security

> For production deployments, set `"restrictToWorkspace": true` in your config to sandbox the agent.

| Option | Default | Description |
|--------|---------|-------------|
| `tools.restrictToWorkspace` | `false` | When `true`, restricts **all** agent tools (shell, file read/write/edit, list) to the workspace directory. Prevents path traversal and out-of-scope access. |
| `channels.*.allowFrom` | `[]` (allow all) | Whitelist of user IDs. Empty = allow everyone; non-empty = only listed users can interact. |


## CLI Reference

| Command | Description |
|---------|-------------|
| `nanobot onboard` | Initialize config & workspace |
| `nanobot agent -m "..."` | Chat with the agent |
| `nanobot agent` | Interactive chat mode |
| `nanobot agent --no-markdown` | Show plain-text replies |
| `nanobot agent --logs` | Show runtime logs during chat |
| `nanobot gateway` | Start the gateway |
| `nanobot status` | Show status |
| `nanobot channels login` | Link WhatsApp (scan QR) |
| `nanobot channels status` | Show channel status |

Interactive mode exits: `exit`, `quit`, `/exit`, `/quit`, `:q`, or `Ctrl+D`.

<details>
<summary><b>Scheduled Tasks (Cron)</b></summary>

```bash
# Add a job
nanobot cron add --name "daily" --message "Good morning!" --cron "0 9 * * *"
nanobot cron add --name "hourly" --message "Check status" --every 3600

# List jobs
nanobot cron list

# Remove a job
nanobot cron remove <job_id>
```

</details>

## 🐳 Docker

> [!TIP]
> The `-v ~/.nanobot:/root/.nanobot` flag mounts your local config directory into the container, so your config and workspace persist across container restarts.

Build and run nanobot in a container:

```bash
# Build the image
docker build -t nanobot .

# Initialize config (first time only)
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot onboard

# Edit config on host to add API keys
vim ~/.nanobot/config.json

# Run gateway (connects to enabled channels, e.g. Telegram/Discord/Mochat)
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway

# Or run a single command
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot agent -m "Hello!"
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot status
```

## 📁 Project Structure

```
nanobot/
├── agent/          # 🧠 Core agent logic
│   ├── loop.py     #    Agent loop (LLM ↔ tool execution)
│   ├── context.py  #    Prompt builder
│   ├── memory.py   #    Persistent memory
│   ├── skills.py   #    Skills loader
│   ├── subagent.py #    Background task execution
│   └── tools/      #    Built-in tools (incl. spawn)
├── skills/         # 🎯 Bundled skills (github, weather, tmux...)
├── channels/       # 📱 Chat channel integrations
├── bus/            # 🚌 Message routing
├── cron/           # ⏰ Scheduled tasks
├── heartbeat/      # 💓 Proactive wake-up
├── providers/      # 🤖 LLM providers (OpenRouter, etc.)
├── session/        # 💬 Conversation sessions
├── config/         # ⚙️ Configuration
└── cli/            # 🖥️ Commands
>>>>>>> origin/feature/video_analysis
```

---

## 🚀 快速开始

### 1. 安装

<<<<<<< HEAD
```bash
pip install -e .
pip install lark-oapi>=1.0.0   # 飞书机器人需要
```
=======
- [x] **Voice Transcription** — Support for Groq Whisper (Issue #13)
- [ ] **Multi-modal** — See and hear (images, voice, video)
- [ ] **Long-term memory** — Never forget important context
- [ ] **Better reasoning** — Multi-step planning and reflection
- [ ] **More integrations** — Calendar and more
- [ ] **Self-improvement** — Learn from feedback and mistakes
>>>>>>> origin/feature/video_analysis

### 2. 配置

<<<<<<< HEAD
参考 [nanobot 文档](https://github.com/HKUDS/nanobot) 配置 LLM Provider（`~/.nanobot/config.json`）。
=======
<a href="https://github.com/HKUDS/nanobot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/nanobot&max=100&columns=12&updated=20260210" alt="Contributors" />
</a>
>>>>>>> origin/feature/video_analysis

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
