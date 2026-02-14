"""Automatic training tracker for PyTorch training loops.

在训练脚本中只需加几行代码，即可自动记录训练过程到 OpenResearchBot 系统，
Agent 和 Dashboard 可以实时看到训练进展。

## 基本用法（PyTorch 原生循环）

```python
from nanobot.tracker import NanobotTracker

# 训练开始 — 自动创建记录、检测 GPU
tracker = NanobotTracker(
    name="OpenVLA-7B finetune",
    model="OpenVLA-7B",
    dataset="bridge_v2",
    hyperparams={"lr": 2e-5, "batch_size": 32, "epochs": 100},
)

for epoch in range(100):
    loss = train_one_epoch()
    tracker.log(epoch=epoch, loss=loss)           # 自动记录指标
    tracker.log(epoch=epoch, loss=val_loss, success_rate=0.8)

tracker.finish()  # 标记完成（也可用 with 语句自动管理）
```

## 使用 with 语句（推荐，异常时自动标记 failed）

```python
with NanobotTracker(name="my-exp", model="OpenVLA-7B") as tracker:
    for epoch in range(100):
        loss = train_one_epoch()
        tracker.log(epoch=epoch, loss=loss)
    # 正常退出自动标记 completed
# 异常退出自动标记 failed
```

## HuggingFace Trainer 集成

```python
from nanobot.tracker import NanobotHFCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[NanobotHFCallback(name="my-exp", model="OpenVLA-7B")],
)
trainer.train()  # 自动记录所有 metrics
```

## 自定义 workspace 路径

```python
tracker = NanobotTracker(name="exp", workspace="/path/to/workspace")
```

默认自动查找: ./workspace > 环境变量 NANOBOT_WORKSPACE > 当前目录
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _short_id() -> str:
    return "run-" + uuid.uuid4().hex[:6]


def _detect_gpu_info() -> str:
    """尝试自动检测 GPU 信息。"""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem
            mem_gb = mem / (1024 ** 3)
            if count > 1:
                return f"{count}x {name} ({mem_gb:.0f}GB)"
            return f"{name} ({mem_gb:.0f}GB)"
    except Exception:
        pass

    # Fallback: try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if lines:
                count = len(lines)
                name, mem = lines[0].split(",", 1)
                mem_gb = int(mem.strip()) / 1024
                if count > 1:
                    return f"{count}x {name.strip()} ({mem_gb:.0f}GB)"
                return f"{name.strip()} ({mem_gb:.0f}GB)"
    except Exception:
        pass

    return ""


def _detect_script_name() -> str:
    """获取当前运行脚本的名称。"""
    try:
        main = sys.modules.get("__main__")
        if main and hasattr(main, "__file__") and main.__file__:
            return Path(main.__file__).stem
    except Exception:
        pass
    return ""


def _find_workspace(explicit: str | Path | None = None) -> Path:
    """查找 workspace 目录。优先级: 显式指定 > 环境变量 > ./workspace > 当前目录"""
    if explicit:
        p = Path(explicit)
        p.mkdir(parents=True, exist_ok=True)
        return p

    env = os.environ.get("NANOBOT_WORKSPACE")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 向上查找 workspace 目录（最多 5 层）
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents)[:5]:
        wp = parent / "workspace"
        if wp.exists() and wp.is_dir():
            return wp

    # 最终 fallback：在当前目录创建 workspace
    wp = cwd / "workspace"
    wp.mkdir(parents=True, exist_ok=True)
    return wp


# ---------------------------------------------------------------------------
# Storage (与 TrainingTrackerTool 完全兼容的 JSON 存储)
# ---------------------------------------------------------------------------

class _Storage:
    """线程安全的 JSON 存储，与 TrainingTrackerTool 共享同一个文件。"""

    def __init__(self, workspace: Path):
        self._path = workspace / "research" / "training_runs.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> list[dict]:
        with self._lock:
            if self._path.exists():
                try:
                    data = json.loads(self._path.read_text(encoding="utf-8"))
                    return data.get("runs", [])
                except (json.JSONDecodeError, KeyError):
                    return []
            return []

    def save(self, runs: list[dict]) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "runs": runs}
            # 写入临时文件再重命名，防止写入中途被读取到不完整数据
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)

    def update_run(self, run_id: str, updater: Any) -> dict | None:
        """读取 → 找到 run → 调用 updater(run) → 保存。返回更新后的 run。"""
        runs = self.load()
        run = next((r for r in runs if r["id"] == run_id), None)
        if run is None:
            return None
        updater(run)
        self.save(runs)
        return run


# ---------------------------------------------------------------------------
# NanobotTracker — PyTorch 训练自动记录器
# ---------------------------------------------------------------------------

class NanobotTracker:
    """PyTorch 训练自动记录器。

    在训练脚本中嵌入几行代码，自动记录训练过程。
    数据直接写入 workspace/research/training_runs.json，
    Agent 和 Dashboard 可以实时看到。

    Features:
    - 自动检测 GPU 信息
    - 自动检测脚本名称
    - with 语句支持（正常退出 → completed，异常 → failed）
    - atexit 兜底（进程被 kill 时标记 stopped）
    - 可控日志频率（log_every_n_steps）
    - 线程安全
    - checkpoint 记录
    """

    def __init__(
        self,
        name: str,
        model: str = "",
        dataset: str = "",
        hyperparams: dict | None = None,
        gpu_info: str = "",
        vla_config: dict | None = None,
        note: str | None = None,
        workspace: str | Path | None = None,
        log_every_n_steps: int = 1,
        auto_detect_gpu: bool = True,
    ):
        """
        Args:
            name: 训练运行的名称（必填）
            model: 模型名称/架构
            dataset: 训练数据集
            hyperparams: 超参数字典
            gpu_info: GPU 信息（留空则自动检测）
            vla_config: VLA 特有配置
            note: 初始备注
            workspace: workspace 路径（留空则自动查找）
            log_every_n_steps: 每 N 次调用 log() 才实际写入磁盘（默认1，即每次都写）
            auto_detect_gpu: 是否自动检测 GPU 信息
        """
        self._workspace = _find_workspace(workspace)
        self._storage = _Storage(self._workspace)
        self._log_every = max(1, log_every_n_steps)
        self._log_count = 0
        self._pending_metrics: dict | None = None  # 缓冲区
        self._finished = False

        # 自动检测 GPU
        if not gpu_info and auto_detect_gpu:
            gpu_info = _detect_gpu_info()

        # 自动补充名称
        if not name:
            name = _detect_script_name() or f"training-{_short_id()}"

        # 自动从 hyperparams 提取常用字段
        hp = hyperparams or {}

        # 创建训练记录
        now = _now_iso()
        self._run_id = _short_id()
        run: dict[str, Any] = {
            "id": self._run_id,
            "name": name,
            "model": model,
            "dataset": dataset,
            "hyperparams": hp,
            "status": "running",
            "gpu_info": gpu_info,
            "vla_config": vla_config or {},
            "metrics_history": [],
            "latest_metrics": {},
            "notes": [],
            "checkpoints": [],
            "created": now,
            "updated": now,
            "started": now,
            "finished": None,
            # 额外元数据
            "_meta": {
                "script": _detect_script_name(),
                "hostname": platform.node(),
                "python": platform.python_version(),
                "pid": os.getpid(),
                "auto_tracked": True,
            },
        }

        if note:
            run["notes"].append({"time": now, "content": note})

        # 保存
        runs = self._storage.load()
        runs.append(run)
        self._storage.save(runs)

        # 注册 atexit，进程意外退出时标记 stopped
        atexit.register(self._atexit_hook)

        self._print(
            f"🚀 NanobotTracker: 训练已注册 [{self._run_id}] {name}"
            + (f" | model={model}" if model else "")
            + (f" | gpu={gpu_info}" if gpu_info else "")
        )

    # ---------- 公开 API ----------

    @property
    def run_id(self) -> str:
        """当前训练运行的 ID。"""
        return self._run_id

    @property
    def workspace(self) -> Path:
        return self._workspace

    def log(self, **metrics: Any) -> None:
        """记录训练指标。

        Examples:
            tracker.log(epoch=1, loss=0.5, lr=1e-4)
            tracker.log(step=1000, loss=0.3, success_rate=0.67)
        """
        if self._finished:
            return

        self._log_count += 1
        self._pending_metrics = metrics

        # 按频率控制写入
        if self._log_count % self._log_every == 0:
            self._flush_metrics()

    def log_checkpoint(self, path: str, **extra_metrics: Any) -> None:
        """记录一个 checkpoint。

        Args:
            path: checkpoint 路径或名称
            **extra_metrics: 可选的额外指标
        """
        if self._finished:
            return

        if extra_metrics:
            self.log(**extra_metrics)

        def _update(run: dict):
            run.setdefault("checkpoints", []).append(path)
            run["updated"] = _now_iso()

        self._storage.update_run(self._run_id, _update)
        self._print(f"💾 Checkpoint: {path}")

    def add_note(self, content: str) -> None:
        """添加一条备注。"""
        if self._finished:
            return

        now = _now_iso()

        def _update(run: dict):
            run.setdefault("notes", []).append({"time": now, "content": content})
            run["updated"] = now

        self._storage.update_run(self._run_id, _update)

    def update_hyperparams(self, **kwargs: Any) -> None:
        """追加/更新超参数。"""
        def _update(run: dict):
            run.setdefault("hyperparams", {}).update(kwargs)
            run["updated"] = _now_iso()

        self._storage.update_run(self._run_id, _update)

    def update_vla_config(self, **kwargs: Any) -> None:
        """追加/更新 VLA 配置。"""
        def _update(run: dict):
            run.setdefault("vla_config", {}).update(kwargs)
            run["updated"] = _now_iso()

        self._storage.update_run(self._run_id, _update)

    def finish(self, note: str | None = None) -> None:
        """标记训练完成。"""
        self._set_status("completed", note)

    def fail(self, note: str | None = None) -> None:
        """标记训练失败。"""
        self._set_status("failed", note)

    def stop(self, note: str | None = None) -> None:
        """标记训练手动停止。"""
        self._set_status("stopped", note)

    # ---------- Context manager ----------

    def __enter__(self) -> "NanobotTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._finished:
            return
        if exc_type is not None:
            # 异常退出 → failed
            err_msg = f"{exc_type.__name__}: {exc_val}" if exc_val else str(exc_type.__name__)
            self.fail(note=f"异常退出: {err_msg}")
        else:
            # 正常退出 → completed
            self.finish()
        return None  # 不抑制异常

    # ---------- 内部方法 ----------

    def _flush_metrics(self) -> None:
        """将缓冲的指标写入磁盘。"""
        metrics = self._pending_metrics
        if not metrics:
            return
        self._pending_metrics = None
        now = _now_iso()
        entry = {"time": now, **metrics}

        def _update(run: dict):
            run.setdefault("metrics_history", []).append(entry)
            run.setdefault("latest_metrics", {}).update(metrics)
            run["updated"] = now
            # 如果还是 queued，自动改成 running
            if run.get("status") == "queued":
                run["status"] = "running"
                if not run.get("started"):
                    run["started"] = now

        self._storage.update_run(self._run_id, _update)

        # 输出进度摘要
        parts = []
        for k in ["epoch", "step", "loss", "success_rate", "lr", "eval_loss"]:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float) and k != "epoch":
                    parts.append(f"{k}={v:.4g}")
                else:
                    parts.append(f"{k}={v}")
        # 补充其余字段
        for k, v in metrics.items():
            if k not in ["epoch", "step", "loss", "success_rate", "lr", "eval_loss"]:
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4g}")
                else:
                    parts.append(f"{k}={v}")
        if parts:
            self._print(f"📈 [{self._run_id}] {', '.join(parts)}")

    def _set_status(self, status: str, note: str | None = None) -> None:
        """设置训练状态并刷新缓冲。"""
        if self._finished:
            return
        self._finished = True

        # 先刷新未写入的指标
        if self._pending_metrics:
            self._flush_metrics()

        now = _now_iso()

        def _update(run: dict):
            run["status"] = status
            run["finished"] = now
            run["updated"] = now
            if note:
                run.setdefault("notes", []).append({"time": now, "content": note})

        self._storage.update_run(self._run_id, _update)

        emoji = {"completed": "✅", "failed": "❌", "stopped": "⏹️"}.get(status, "❓")
        self._print(f"{emoji} [{self._run_id}] 训练{status}")

    def _atexit_hook(self) -> None:
        """进程退出时的兜底处理。"""
        if not self._finished:
            # 刷新缓冲指标
            if self._pending_metrics:
                self._flush_metrics()
            self._set_status("stopped", note="进程退出 (atexit)")

    @staticmethod
    def _print(msg: str) -> None:
        """安全打印（兼容 Windows GBK 终端）。"""
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(
                msg.encode(sys.stdout.encoding or "utf-8", errors="replace")
                .decode(sys.stdout.encoding or "utf-8", errors="replace"),
                flush=True,
            )


# ---------------------------------------------------------------------------
# NanobotHFCallback — HuggingFace Trainer 回调
# ---------------------------------------------------------------------------

class NanobotHFCallback:
    """HuggingFace Transformers Trainer 自动记录回调。

    Usage:
        from nanobot.tracker import NanobotHFCallback

        trainer = Trainer(
            model=model,
            args=training_args,
            callbacks=[NanobotHFCallback(name="my-exp", model="OpenVLA-7B")],
        )
        trainer.train()
    """

    def __init__(
        self,
        name: str = "",
        model: str = "",
        dataset: str = "",
        hyperparams: dict | None = None,
        gpu_info: str = "",
        vla_config: dict | None = None,
        workspace: str | Path | None = None,
        log_every_n_steps: int = 1,
    ):
        self._init_kwargs = {
            "name": name,
            "model": model,
            "dataset": dataset,
            "hyperparams": hyperparams,
            "gpu_info": gpu_info,
            "vla_config": vla_config,
            "workspace": workspace,
            "log_every_n_steps": log_every_n_steps,
        }
        self._tracker: NanobotTracker | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时自动创建记录。"""
        init = self._init_kwargs.copy()

        # 从 TrainingArguments 提取超参数
        hp = init.get("hyperparams") or {}
        if hasattr(args, "learning_rate"):
            hp.setdefault("lr", args.learning_rate)
        if hasattr(args, "per_device_train_batch_size"):
            hp.setdefault("batch_size", args.per_device_train_batch_size)
        if hasattr(args, "num_train_epochs"):
            hp.setdefault("epochs", int(args.num_train_epochs))
        if hasattr(args, "weight_decay"):
            hp.setdefault("weight_decay", args.weight_decay)
        if hasattr(args, "warmup_steps"):
            hp.setdefault("warmup_steps", args.warmup_steps)
        if hasattr(args, "gradient_accumulation_steps"):
            hp.setdefault("grad_accum", args.gradient_accumulation_steps)
        init["hyperparams"] = hp

        # 从 model kwargs 提取模型名称
        if not init["name"]:
            init["name"] = _detect_script_name() or "hf-training"
        if not init["model"] and "model" in kwargs:
            m = kwargs["model"]
            if hasattr(m, "config") and hasattr(m.config, "_name_or_path"):
                init["model"] = m.config._name_or_path

        self._tracker = NanobotTracker(**init)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次 Trainer 打印日志时自动记录指标。"""
        if self._tracker is None or logs is None:
            return

        # 提取有用的指标
        metrics = {}
        for key in ["loss", "eval_loss", "learning_rate", "epoch",
                     "grad_norm", "eval_accuracy", "eval_f1",
                     "eval_precision", "eval_recall"]:
            if key in logs:
                metrics[key] = logs[key]

        # 记录 step
        if state and hasattr(state, "global_step"):
            metrics["step"] = state.global_step

        # 也记录其他 eval_ 开头的指标
        for k, v in logs.items():
            if k.startswith("eval_") and k not in metrics:
                metrics[k] = v

        if metrics:
            self._tracker.log(**metrics)

    def on_save(self, args, state, control, **kwargs):
        """保存 checkpoint 时自动记录。"""
        if self._tracker is None:
            return
        if state and hasattr(state, "best_model_checkpoint") and state.best_model_checkpoint:
            self._tracker.log_checkpoint(state.best_model_checkpoint)
        elif args and hasattr(args, "output_dir"):
            step = state.global_step if state else "unknown"
            self._tracker.log_checkpoint(f"{args.output_dir}/checkpoint-{step}")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时自动标记完成。"""
        if self._tracker is None:
            return
        self._tracker.finish()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估结束时记录评估指标。"""
        if self._tracker is None or metrics is None:
            return
        eval_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                eval_metrics[k] = v
        if eval_metrics:
            self._tracker.log(**eval_metrics)


# ---------------------------------------------------------------------------
# PyTorch Lightning Callback (可选)
# ---------------------------------------------------------------------------

class NanobotLightningCallback:
    """PyTorch Lightning 自动记录回调。

    Usage:
        from nanobot.tracker import NanobotLightningCallback

        trainer = pl.Trainer(
            callbacks=[NanobotLightningCallback(name="my-exp", model="OpenVLA-7B")],
        )
    """

    def __init__(
        self,
        name: str = "",
        model: str = "",
        dataset: str = "",
        hyperparams: dict | None = None,
        workspace: str | Path | None = None,
        log_every_n_steps: int = 1,
    ):
        self._init_kwargs = {
            "name": name or _detect_script_name() or "lightning-training",
            "model": model,
            "dataset": dataset,
            "hyperparams": hyperparams,
            "workspace": workspace,
            "log_every_n_steps": log_every_n_steps,
        }
        self._tracker: NanobotTracker | None = None

    def on_train_start(self, trainer, pl_module):
        hp = self._init_kwargs.get("hyperparams") or {}
        if hasattr(trainer, "max_epochs"):
            hp.setdefault("epochs", trainer.max_epochs)
        if hasattr(pl_module, "learning_rate"):
            hp.setdefault("lr", pl_module.learning_rate)
        self._init_kwargs["hyperparams"] = hp
        self._tracker = NanobotTracker(**self._init_kwargs)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._tracker is None:
            return
        metrics = {"epoch": trainer.current_epoch}
        # 从 trainer.callback_metrics 获取
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                pass
        self._tracker.log(**metrics)

    def on_train_end(self, trainer, pl_module):
        if self._tracker:
            self._tracker.finish()


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def track_training(
    name: str,
    model: str = "",
    dataset: str = "",
    hyperparams: dict | None = None,
    **kwargs: Any,
) -> NanobotTracker:
    """便捷函数，等同于 NanobotTracker(...)。

    Example:
        tracker = track_training("my-exp", model="OpenVLA-7B", hyperparams={"lr": 2e-5})
    """
    return NanobotTracker(
        name=name, model=model, dataset=dataset,
        hyperparams=hyperparams, **kwargs,
    )
