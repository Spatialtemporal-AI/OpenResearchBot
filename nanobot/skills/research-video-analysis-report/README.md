# 科研视频分析报告 Skill

一个专为科研工作者和工程师设计的视频分析工具，使用 Google Gemini 的 Files API 进行直接视频理解，自动生成结构化的 Markdown 分析报告。

## 功能特性

- 🎥 **支持多种输入格式**：本地视频文件或网络视频 URL
- 🤖 **智能领域推断**：自动识别视频所属的科研领域（机器人学、计算机视觉、物理实验等）
- 📊 **结构化报告生成**：生成包含时间线、关键观察、风险分析和可执行建议的科研级报告
- 🔄 **自动视频预处理**：自动检测并转换不兼容的视频格式，确保上传成功
- 💾 **会话管理**：每次分析创建独立的会话目录，保存完整的分析过程数据
- 🧹 **自动清理**：默认自动清理临时文件和远程上传文件，节省存储空间

## 系统要求

### 必需依赖

- Python 3.8+
- `google-genai` SDK
- `ffmpeg` 和 `ffprobe`（用于视频处理）

### 安装依赖

```bash
# 安装 Python SDK
pip install google-genai requests

# 安装 ffmpeg（macOS）
brew install ffmpeg

# 安装 ffmpeg（Ubuntu/Debian）
sudo apt-get install ffmpeg

# 安装 ffmpeg（Windows）
# 从 https://ffmpeg.org/download.html 下载并添加到 PATH
```

## 快速开始

### 1. 配置 API 密钥

设置 Gemini API 密钥（三种方式，按优先级排序）：

**方式 1：单次命令临时注入**
```bash
GEMINI_API_KEY="YOUR_KEY" nanobot agent -m "请使用 research-video-analysis-report skill 分析视频，input=video.mp4。"
```

**方式 2：环境变量**
```bash
export GEMINI_API_KEY="YOUR_KEY"
```

**方式 3：配置文件**
在 `~/.nanobot/config.json` 中配置：
```json
{
  "providers": {
    "gemini": {
      "apiKey": "YOUR_KEY"
    }
  }
}
```

### 2. 基本使用

**方式 A：进入交互模式（推荐）**
```bash
nanobot agent
```

进入交互后输入：
```text
请使用 research-video-analysis-report skill 分析视频：
- input: /path/to/experiment_video.mp4
- task: 分析机器人抓取任务的执行情况
- notes: 使用 Kinova Gen3 机械臂，目标物体为 5 个不同形状的物体
```
或者直接就说：
```text
请使用 视频分析 skill 分析视频/path/to/experiment_video.mp4并生成报告。
```

**方式 B：单次命令模式（`-m`）**
```bash
nanobot agent -m "请使用 research-video-analysis-report skill 分析视频，input=https://example.com/research_video.mp4，task=总结关键事件和风险。"
```

## 详细使用说明

### 命令行参数

#### 必需参数

- `--input`: 视频输入路径（本地文件路径或 HTTP/HTTPS URL）

#### 可选参数

**任务描述：**
- `--task`: 任务目标描述（例如："分析实验目标和成功标准"）
- `--notes`: 相关备注信息（例如："设置细节、模型版本、异常情况"）

**输出配置：**
- `--output-root`: 输出根目录（默认：`./output`）
- `--session-id`: 自定义会话 ID（默认：使用时间戳）
- `--report-name`: 报告文件名（默认：`report.md`）

**模型配置：**
- `--model`: Gemini 模型 ID（默认：`gemini-3-flash-preview`）
- `--api-key`: Gemini API 密钥（可选，优先使用环境变量或配置文件）
- `--max-output-tokens`: 最大输出 token 数（默认：4096）

**视频处理：**
- `--size-threshold-mb`: 压缩阈值（MB，默认：100.0）
  - 超过此大小的视频会被自动压缩
- `--force-compress`: 强制压缩视频（即使未超过阈值）
- `--compression-crf`: 压缩质量（0-51，默认：20，数值越小质量越高）
  - 推荐值：18（高质量）、20（默认）、23（平衡）
- `--compression-preset`: 压缩速度预设（默认：`medium`）
  - 选项：`ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`
  - 速度越慢，压缩效率越高

**高级选项：**
- `--prompt-file`: 自定义 prompt 模板文件路径
- `--dry-run`: 仅预处理视频，不调用 Gemini API
- `--keep-cache`: 保留临时缓存文件
- `--keep-remote-file`: 保留上传到 Gemini 的远程文件
- `--max-wait-seconds`: 等待文件处理的最大秒数（默认：300）
- `--poll-interval-seconds`: 轮询文件状态的间隔（默认：3.0 秒）

### 使用示例

**高质量分析（大文件）：**
```bash
nanobot agent -m "请使用 research-video-analysis-report skill 分析视频，input=/path/to/large_video.avi，task=详细分析机器人抓取策略。视频较大，请使用 size-threshold-mb=100、compression-crf=18、compression-preset=slow。"
```

**使用自定义 prompt：**
```bash
nanobot agent -m "请使用 research-video-analysis-report skill 分析视频，input=/path/to/video.mp4，prompt-file=/path/to/custom_prompt.md。"
```

**保留所有中间文件（用于调试）：**
```bash
nanobot agent -m "请使用 research-video-analysis-report skill 分析视频，input=/path/to/video.mp4，并启用 keep-cache=true、keep-remote-file=true。"
```

## 输出说明

### 报告结构

生成的报告遵循 `references/report-schema.md` 中定义的标准格式，包含以下部分：

1. **摘要 (Summary)**
   - 视频内容概述
   - 主要发现

2. **领域判断与置信度 (Inferred Domain & Task)**
   - 推断的科研领域
   - 任务类型
   - 置信度评估

3. **假设与设置 (Assumptions / Setup)**
   - 从视频中可以推断的信息
   - 无法确定的信息

4. **关键事件时间线 (Timeline of Key Events)**
   - 带时间戳的事件序列
   - 表格形式展示

5. **关键观察 (Key Observations)**
   - 领域特定的观察结果
   - 技术细节分析

6. **失败/风险分析 (Failure / Risk Analysis)**
   - 潜在问题识别
   - 风险评估
   - 证据说明

7. **可执行建议 (Actionable Recommendations)**
   - 数据收集/标注建议
   - 模型/策略改进建议
   - 方法/系统优化建议
   - 评估指标建议

8. **下一步实验计划 (Next Experiment Plan)**
   - 3-6 个实验建议
   - 每个实验的通过/失败标准

9. **附录 (Appendix)**
   - 视频元数据摘要
   - 模型参数
   - 已知限制

### 输出目录结构

每次分析会在 `output/<session-id>/` 目录下创建以下文件：

```
output/20260211T180159+0800/
├── report.md                    # 生成的分析报告（主要输出）
├── prompt.txt                   # 发送给模型的完整 prompt
├── manifest.json                # 运行元数据和参数记录
├── source_metadata.json         # 原始视频的元数据（ffprobe）
├── prepared_metadata.json       # 预处理后视频的元数据
├── upload_response.json         # Gemini Files API 上传响应
└── model_response.json          # Gemini 模型的完整响应
```

## 视频处理逻辑

### 自动预处理触发条件

脚本会在以下情况下自动预处理视频：

1. **容器格式不是 MP4**
2. **视频编码不是 H.264**
3. **文件大小超过阈值**（默认 100 MB）
4. **用户强制压缩**（`--force-compress`）

### 预处理过程

1. 使用 `ffmpeg` 将视频转换为 H.264 编码的 MP4 格式
2. 保持原始分辨率和时间结构
3. 使用高质量压缩设置（默认 CRF=20）
4. 添加快速启动标志（`faststart`）以优化流媒体播放

### 质量建议

- **默认设置（CRF=20）**：适合大多数场景，平衡质量和文件大小
- **高质量（CRF=18）**：适合需要精细分析的场景，文件较大
- **快速处理（preset=fast）**：适合快速预览，质量略低
- **最佳质量（preset=slow, CRF=18）**：适合最终分析，处理时间较长

## 工作流程

1. **收集运行上下文**
   - 询问任务目标、成功标准和已知异常
   - 获取视频输入（本地路径或 URL）

2. **准备视频**
   - 如果是 URL，先下载到会话缓存
   - 检查视频格式和大小
   - 必要时进行转码和压缩

3. **上传到 Gemini**
   - 使用 `google-genai` SDK 上传视频文件
   - 等待文件处理完成（ACTIVE 状态）

4. **生成报告**
   - 调用 Gemini 模型进行视频分析
   - 使用领域自适应的 prompt 模板
   - 模型自动推断领域和任务类型

5. **保存结果**
   - 生成 Markdown 报告
   - 保存所有中间数据和元数据

6. **清理（默认）**
   - 删除临时缓存文件
   - 删除远程上传的文件

## 故障排除

### 常见问题

**1. 缺少 ffmpeg/ffprobe**
```
Error: Missing required binary: ffmpeg
```
**解决方案**：安装 ffmpeg（见"系统要求"部分）

**2. API 密钥未设置**
```
Error: missing API key
```
**解决方案**：设置 `GEMINI_API_KEY` 环境变量或使用 `--api-key` 参数

**3. 视频上传超时**
```
Error: Timed out waiting for uploaded file to become ACTIVE/READY
```
**解决方案**：
- 增加 `--max-wait-seconds` 参数值
- 检查网络连接
- 尝试压缩视频（使用 `--force-compress`）

**4. 视频格式不支持**
```
Error: Command failed (ffmpeg ...)
```
**解决方案**：
- 确保 ffmpeg 已正确安装
- 检查视频文件是否损坏
- 尝试手动转换视频格式

**5. 内存不足**
```
Error: Out of memory
```
**解决方案**：
- 降低 `--compression-crf` 值（增加压缩）
- 减小 `--size-threshold-mb` 值（更早触发压缩）
- 使用 `--compression-preset fast` 加快处理

## 高级用法

### 自定义 Prompt 模板

创建自定义 prompt 文件，使用 `{{TASK}}` 和 `{{NOTES}}` 作为占位符：

```markdown
# 自定义分析 Prompt

分析目标：{{TASK}}

备注信息：{{NOTES}}

请按照以下要求生成报告：
1. 重点关注...
2. 详细分析...
```

然后使用 `--prompt-file` 参数指定该文件。

### 批量处理

使用 shell 脚本批量处理多个视频：

```bash
#!/bin/bash
for video in /path/to/videos/*.mp4; do
    nanobot agent -m "请使用 research-video-analysis-report skill 分析视频，input=$video，task=批量分析实验视频，output-root=./batch_output。"
done
```

### 集成到工作流

可将单次命令模式集成到其他工具中，并重定向输出：

```bash
nanobot agent -m "请使用 research-video-analysis-report skill 分析视频，input=video.mp4，task=生成可归档报告。" > analysis_result.md
```

## 技术细节

### 使用的技术栈

- **Google Gemini API**：视频理解和内容生成
- **google-genai SDK**：官方 Python SDK，内部使用 Files API
- **ffmpeg**：视频转码和压缩
- **requests**：HTTP 视频下载

### 文件上传流程

1. 使用 `client.files.upload()` 上传视频文件
2. 轮询文件状态直到变为 `ACTIVE`
3. 使用 `client.models.generate_content()` 进行分析
4. 默认自动删除远程文件（除非使用 `--keep-remote-file`）

### 会话管理

- 每个分析会话使用唯一的时间戳 ID（格式：`YYYYMMDDTHHMMSS+0800`）
- 所有相关文件保存在独立的会话目录中
- 支持自定义会话 ID（`--session-id`）

## 参考文档

- `SKILL.md`：Skill 的完整说明文档
- `references/report-schema.md`：报告格式规范
- `references/router-schema.md`：可选的未来两阶段路由方案
- `assets/prompt_research_assistant.md`：默认 prompt 模板

## 许可证

本 Skill 遵循项目主许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个 Skill。

## 更新日志

- **初始版本**：支持本地和 URL 视频输入，自动视频预处理，结构化报告生成
