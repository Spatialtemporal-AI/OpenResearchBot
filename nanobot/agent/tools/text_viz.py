"""Pure-text visualization helpers for tracker tools.

All output is plain Unicode text that renders correctly in:
- Chat interfaces (Cursor IDE, web chat, mobile)
- Terminals / consoles
- Markdown renderers

No external dependencies required.
"""

from __future__ import annotations

import math
from typing import Sequence


# ── Building blocks ──────────────────────────────────────────────

BLOCK_FULL = "█"
BLOCK_LIGHT = "░"
BULLET = "●"
DIAMOND = "◆"
TRIANGLE = "▲"
SQUARE = "■"
CROSS = "✕"
DASH_LINE = "─"
VERT_LINE = "│"
CORNER_BL = "└"
CORNER_TL = "┌"
CORNER_TR = "┐"
CORNER_BR = "┘"
TEE_LEFT = "├"
TEE_RIGHT = "┤"
TEE_TOP = "┬"
TEE_BOTTOM = "┴"

MARKERS = [BULLET, DIAMOND, TRIANGLE, SQUARE, CROSS]


def _fmt_num(v: float, width: int = 6) -> str:
    """Format a number to fit in *width* characters."""
    if v == 0:
        return "0".rjust(width)
    abs_v = abs(v)
    if abs_v >= 1000:
        return f"{v:.0f}".rjust(width)
    if abs_v >= 100:
        return f"{v:.1f}".rjust(width)
    if abs_v >= 10:
        return f"{v:.2f}".rjust(width)
    if abs_v >= 1:
        return f"{v:.3f}".rjust(width)
    # < 1  – show significant digits
    return f"{v:.4f}".rjust(width)


# ── Bar chart ────────────────────────────────────────────────────

def bar_chart(
    labels: Sequence[str],
    values: Sequence[float | int],
    *,
    title: str = "",
    max_bar_width: int = 25,
    show_values: bool = True,
) -> str:
    """Horizontal bar chart.

    Example output::

        📊 Task Status
        ──────────────────────────
        todo     ████████░░░░░  3
        doing    ████░░░░░░░░░  2
        done     █████████████  5
        blocked  ██░░░░░░░░░░░  1
    """
    if not labels or not values:
        return "(no data)"

    max_val = max(values) if values else 1
    if max_val == 0:
        max_val = 1
    label_w = max(len(str(l)) for l in labels)

    lines: list[str] = []
    if title:
        lines.append(f"📊 {title}")
        lines.append(DASH_LINE * (label_w + max_bar_width + 10))

    for label, val in zip(labels, values):
        filled = round(val / max_val * max_bar_width)
        empty = max_bar_width - filled
        bar = BLOCK_FULL * filled + BLOCK_LIGHT * empty
        val_str = f"  {val}" if show_values else ""
        lines.append(f"{str(label).ljust(label_w)}  {bar}{val_str}")

    return "\n".join(lines)


# ── Sparkline (single row) ──────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float | int], *, label: str = "") -> str:
    """Compact one-line sparkline.

    Example: ``loss  ▇▆▅▄▃▂▂▁  0.19``
    """
    if not values:
        return f"{label}  (no data)" if label else "(no data)"

    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    chars = []
    for v in values:
        idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])

    spark = "".join(chars)
    last_val = _fmt_num(values[-1]).strip()
    prefix = f"{label}  " if label else ""
    return f"{prefix}{spark}  {last_val}"


# ── Line chart (multi-row) ──────────────────────────────────────

def line_chart(
    x_values: Sequence[str | int | float],
    y_series: dict[str, Sequence[float | int]],
    *,
    title: str = "",
    height: int = 10,
    width: int = 40,
) -> str:
    """ASCII line chart supporting multiple series.

    Parameters
    ----------
    x_values : labels for the x-axis
    y_series : ``{"series_name": [y0, y1, ...], ...}``
    title    : chart title
    height   : number of text rows for the plot area
    width    : number of text columns for the plot area
    """
    if not y_series:
        return "(no data)"

    # Flatten all y-values to find global range
    all_y: list[float] = []
    for vs in y_series.values():
        all_y.extend(float(v) for v in vs)
    if not all_y:
        return "(no data)"

    y_min, y_max = min(all_y), max(all_y)
    if y_min == y_max:
        y_max = y_min + 1  # avoid division by zero

    # Determine actual width (at least number of points)
    n_points = max(len(vs) for vs in y_series.values())
    plot_w = max(width, n_points)

    # Y-axis label width
    y_label_w = 7  # e.g. " 0.850"

    # Build empty canvas
    canvas: list[list[str]] = [[" " for _ in range(plot_w)] for _ in range(height)]

    # Plot each series
    series_names = list(y_series.keys())
    for si, (sname, ys) in enumerate(y_series.items()):
        marker = MARKERS[si % len(MARKERS)]
        for i, y in enumerate(ys):
            # Map data index to canvas column
            if n_points == 1:
                cx = plot_w // 2
            else:
                cx = round(i / (n_points - 1) * (plot_w - 1))
            # Map y to canvas row (top = height-1 = y_max, bottom = 0 = y_min)
            cy = round((float(y) - y_min) / (y_max - y_min) * (height - 1))
            cy = max(0, min(height - 1, cy))
            row_idx = height - 1 - cy  # invert for top-to-bottom
            canvas[row_idx][cx] = marker

    # Assemble output
    lines: list[str] = []
    if title:
        lines.append(f"📈 {title}")
        lines.append(DASH_LINE * (y_label_w + 2 + plot_w + 2))

    for row_i, row in enumerate(canvas):
        # Y label for top, middle, bottom
        if row_i == 0:
            y_lbl = _fmt_num(y_max, y_label_w)
        elif row_i == height - 1:
            y_lbl = _fmt_num(y_min, y_label_w)
        elif row_i == height // 2:
            y_lbl = _fmt_num((y_min + y_max) / 2, y_label_w)
        else:
            y_lbl = " " * y_label_w
        lines.append(f"{y_lbl} {VERT_LINE}{''.join(row)}")

    # X axis
    x_axis = " " * y_label_w + " " + CORNER_BL + DASH_LINE * plot_w
    lines.append(x_axis)

    # X labels (first and last)
    if x_values:
        first_lbl = str(x_values[0])
        last_lbl = str(x_values[-1]) if len(x_values) > 1 else ""
        padding = plot_w - len(first_lbl) - len(last_lbl)
        if padding < 1:
            padding = 1
        x_label_line = " " * (y_label_w + 2) + first_lbl + " " * padding + last_lbl
        lines.append(x_label_line)

    # Legend
    if len(series_names) > 1:
        legend_parts = []
        for si, sn in enumerate(series_names):
            m = MARKERS[si % len(MARKERS)]
            legend_parts.append(f"  {m} {sn}")
        lines.append("".join(legend_parts))

    return "\n".join(lines)


# ── Progress bar ─────────────────────────────────────────────────

def progress_bar(
    current: float,
    total: float,
    *,
    label: str = "",
    width: int = 25,
) -> str:
    """Simple progress bar.

    Example: ``Epoch  [████████████░░░░░░░░░░░░░]  48%``
    """
    if total <= 0:
        ratio = 0.0
    else:
        ratio = min(current / total, 1.0)
    filled = round(ratio * width)
    empty = width - filled
    pct = f"{ratio * 100:.0f}%"
    bar = BLOCK_FULL * filled + BLOCK_LIGHT * empty
    prefix = f"{label}  " if label else ""
    return f"{prefix}[{bar}]  {pct}"


# ── Metric table ─────────────────────────────────────────────────

def metric_table(
    rows: Sequence[dict[str, str | int | float]],
    columns: Sequence[str],
    *,
    title: str = "",
) -> str:
    """Formatted ASCII table.

    Example::

        ┌──────┬────────┬──────┬──────────────┐
        │ epoch│   loss │  mse │ success_rate │
        ├──────┼────────┼──────┼──────────────┤
        │   10 │  0.850 │0.045 │          -   │
        │   30 │  0.420 │0.023 │          -   │
        │   50 │  0.280 │0.015 │       0.62   │
        └──────┴────────┴──────┴──────────────┘
    """
    if not rows or not columns:
        return "(no data)"

    # Compute column widths
    col_widths: dict[str, int] = {}
    for c in columns:
        col_widths[c] = len(c)
        for r in rows:
            val = r.get(c, "-")
            col_widths[c] = max(col_widths[c], len(str(val)))

    def _row_str(values: dict[str, str | int | float]) -> str:
        cells = []
        for c in columns:
            val = str(values.get(c, "-"))
            cells.append(val.rjust(col_widths[c]))
        return VERT_LINE + VERT_LINE.join(f" {cell} " for cell in cells) + VERT_LINE

    def _border(left: str, mid: str, right: str) -> str:
        parts = [DASH_LINE * (col_widths[c] + 2) for c in columns]
        return left + mid.join(parts) + right

    lines: list[str] = []
    if title:
        lines.append(f"  {title}")

    lines.append(_border(CORNER_TL, TEE_TOP, CORNER_TR))
    # Header
    header_vals = {c: c for c in columns}
    lines.append(_row_str(header_vals))
    lines.append(_border(TEE_LEFT, "┼", TEE_RIGHT))
    # Data rows
    for r in rows:
        lines.append(_row_str(r))
    lines.append(_border(CORNER_BL, TEE_BOTTOM, CORNER_BR))

    return "\n".join(lines)


# ── Dashboard (composite) ───────────────────────────────────────

def training_dashboard(
    run: dict,
    *,
    chart_height: int = 8,
    chart_width: int = 40,
) -> str:
    """Full ASCII dashboard for a single training run.

    Combines: header, progress bar, metrics table, sparklines, line chart.
    """
    name = run.get("name", "unnamed")
    run_id = run.get("id", "?")
    status = run.get("status", "?")
    model = run.get("model", "?")

    status_icon = {
        "queued": "⏳", "running": "🔄", "completed": "✅",
        "failed": "❌", "stopped": "⏹️",
    }.get(status, "❓")

    lines: list[str] = []

    # ── Header ──
    inner_w = 54
    border = "═" * inner_w
    lines.append(f"╔{border}╗")

    def _pad_line(text: str) -> str:
        """Pad a line to fit inside the box (accounting for wide chars)."""
        # Simple approach: pad to inner_w visible characters
        # We can't perfectly handle CJK / emoji widths in plain text,
        # so we pad generously.
        padding = inner_w - 2 - len(text)
        if padding < 0:
            text = text[: inner_w - 2]
            padding = 0
        return f"║ {text}{' ' * padding} ║"

    lines.append(_pad_line(f"{status_icon} {name[:48]}"))
    lines.append(_pad_line(f"ID: {run_id}  |  Model: {model}"))
    lines.append(_pad_line(f"Status: {status}"))
    lines.append(f"╚{border}╝")

    # ── Progress bar (if epoch info available) ──
    mh = run.get("metrics_history", [])
    hp = run.get("hyperparams", {})
    total_epochs = hp.get("epochs", 0)
    if mh and total_epochs:
        last_epoch = 0
        for entry in mh:
            if "epoch" in entry:
                last_epoch = max(last_epoch, entry["epoch"])
        lines.append("")
        lines.append(progress_bar(last_epoch, total_epochs, label="Epoch", width=30))
        lines.append(f"  {last_epoch} / {total_epochs} epochs")

    # ── Metrics history table ──
    if mh:
        # Collect all metric keys (exclude 'time')
        all_keys: list[str] = []
        for entry in mh:
            for k in entry:
                if k != "time" and k not in all_keys:
                    all_keys.append(k)

        # Preferred ordering
        preferred = ["epoch", "step", "loss", "eval_loss", "action_mse", "success_rate"]
        ordered_keys: list[str] = [k for k in preferred if k in all_keys]
        ordered_keys += [k for k in all_keys if k not in ordered_keys]

        table_rows: list[dict] = []
        for entry in mh:
            row: dict[str, str | int | float] = {}
            for k in ordered_keys:
                val = entry.get(k)
                if val is not None:
                    row[k] = val
                else:
                    row[k] = "-"
            table_rows.append(row)

        lines.append("")
        lines.append(metric_table(table_rows, ordered_keys, title="📋 Metrics History"))

    # ── Sparklines for key metrics ──
    if mh and len(mh) >= 2:
        lines.append("")
        lines.append("📉 Trends (sparklines)")
        lines.append(DASH_LINE * 40)

        for key in ["loss", "eval_loss", "action_mse", "success_rate"]:
            vals = [entry[key] for entry in mh if key in entry]
            if len(vals) >= 2:
                lines.append(sparkline(vals, label=key.ljust(14)))

    # ── Line chart for loss/success_rate ──
    chartable_series: dict[str, list[float]] = {}
    x_labels: list[str] = []
    if mh and len(mh) >= 2:
        for key in ["loss", "success_rate", "action_mse"]:
            vals = [entry[key] for entry in mh if key in entry]
            if len(vals) >= 2:
                chartable_series[key] = vals

        # X labels from epochs or steps
        if any("epoch" in e for e in mh):
            x_labels = [str(e.get("epoch", "?")) for e in mh if "epoch" in e]
        elif any("step" in e for e in mh):
            x_labels = [str(e.get("step", "?")) for e in mh if "step" in e]

    if chartable_series:
        lines.append("")
        lines.append(line_chart(
            x_labels or list(range(len(next(iter(chartable_series.values()))))),
            chartable_series,
            title="Training Curves",
            height=chart_height,
            width=chart_width,
        ))

    # ── VLA config ──
    vla = run.get("vla_config", {})
    if vla:
        lines.append("")
        lines.append("🤖 VLA Configuration")
        lines.append(DASH_LINE * 40)
        label_map = {
            "action_space": "Action Space",
            "observation_space": "Obs Space",
            "embodiment": "Embodiment",
            "environment": "Environment",
            "task_suite": "Task Suite",
            "action_tokenizer": "Tokenizer",
            "backbone": "Backbone",
        }
        for k, v in vla.items():
            display_k = label_map.get(k, k)
            lines.append(f"  {display_k:<14}: {v}")

    # ── Checkpoints ──
    cps = run.get("checkpoints", [])
    if cps:
        lines.append("")
        lines.append(f"💾 Checkpoints ({len(cps)})")
        for cp in cps[-5:]:
            lines.append(f"  • {cp}")

    # ── Notes ──
    notes = run.get("notes", [])
    if notes:
        lines.append("")
        lines.append(f"📝 Notes ({len(notes)})")
        for n in notes[-5:]:
            t = n.get("time", "")
            lines.append(f"  [{t}] {n.get('content', '')}")

    return "\n".join(lines)


def task_dashboard(tasks: list[dict]) -> str:
    """Full ASCII dashboard for task overview.

    Combines: status bar chart, priority breakdown, task list by status.
    """
    if not tasks:
        return "📋 No tasks to display."

    lines: list[str] = []

    inner_w = 54
    border = "═" * inner_w
    lines.append(f"╔{border}╗")
    title_text = "📋 Research Task Dashboard"
    pad = inner_w - 2 - len(title_text)
    lines.append(f"║ {title_text}{' ' * max(pad, 0)} ║")
    lines.append(f"╚{border}╝")

    # ── Status distribution bar chart ──
    status_counts: dict[str, int] = {}
    for t in tasks:
        s = t.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    status_order = ["doing", "todo", "blocked", "done"]
    status_emoji = {"doing": "🔵", "todo": "⚪", "blocked": "🔴", "done": "✅"}
    bar_labels = []
    bar_values = []
    for s in status_order:
        if s in status_counts:
            bar_labels.append(f"{status_emoji.get(s, '')} {s}")
            bar_values.append(status_counts[s])

    lines.append("")
    lines.append(bar_chart(bar_labels, bar_values, title="Status Distribution", max_bar_width=20))

    # ── Priority distribution (active tasks only) ──
    active = [t for t in tasks if t.get("status") != "done"]
    if active:
        prio_counts: dict[str, int] = {}
        for t in active:
            p = t.get("priority", "medium")
            prio_counts[p] = prio_counts.get(p, 0) + 1

        prio_emoji = {"high": "🔥", "medium": "➖", "low": "💤"}
        prio_labels = []
        prio_values = []
        for p in ["high", "medium", "low"]:
            if p in prio_counts:
                prio_labels.append(f"{prio_emoji.get(p, '')} {p}")
                prio_values.append(prio_counts[p])

        lines.append("")
        lines.append(bar_chart(prio_labels, prio_values, title="Priority (active tasks)", max_bar_width=20))

    # ── Completion progress ──
    total = len(tasks)
    done_count = status_counts.get("done", 0)
    lines.append("")
    lines.append(progress_bar(done_count, total, label="Completion", width=30))
    lines.append(f"  {done_count} / {total} tasks done")

    # ── Task list grouped by status ──
    lines.append("")
    lines.append(DASH_LINE * 56)

    for s in status_order:
        group = [t for t in tasks if t.get("status") == s]
        if not group:
            continue
        emoji = status_emoji.get(s, "")
        lines.append(f"\n{emoji} {s.upper()} ({len(group)})")

        # Sort by priority within group
        prio_rank = {"high": 0, "medium": 1, "low": 2}
        group.sort(key=lambda t: prio_rank.get(t.get("priority", "medium"), 1))

        for t in group:
            prio_icon = {"high": "🔥", "medium": "  ", "low": "💤"}.get(t.get("priority", ""), "  ")
            tag_str = f" [{', '.join(t.get('tags', []))}]" if t.get("tags") else ""
            lines.append(f"  {prio_icon} [{t['id']}] {t['title']}{tag_str}")

    return "\n".join(lines)


def compare_dashboard(
    runs: list[dict],
    *,
    chart_height: int = 8,
    chart_width: int = 40,
) -> str:
    """Visual comparison of multiple training runs."""
    if len(runs) < 2:
        return "Need at least 2 runs to compare."

    lines: list[str] = []

    inner_w = 54
    border = "═" * inner_w
    lines.append(f"╔{border}╗")
    title_text = "📊 Training Run Comparison"
    pad = inner_w - 2 - len(title_text)
    lines.append(f"║ {title_text}{' ' * max(pad, 0)} ║")
    lines.append(f"╚{border}╝")

    # ── Summary table ──
    columns = ["run", "model", "status"]
    # Collect all metric keys
    all_metric_keys: list[str] = []
    for r in runs:
        for k in r.get("latest_metrics", {}):
            if k not in all_metric_keys:
                all_metric_keys.append(k)

    preferred = ["epoch", "step", "loss", "eval_loss", "action_mse", "success_rate"]
    ordered_metrics = [k for k in preferred if k in all_metric_keys]
    ordered_metrics += [k for k in all_metric_keys if k not in ordered_metrics]

    columns.extend(ordered_metrics)

    table_rows: list[dict] = []
    for r in runs:
        row: dict[str, str | int | float] = {
            "run": r.get("name", r.get("id", "?"))[:18],
            "model": r.get("model", "?")[:14],
            "status": r.get("status", "?"),
        }
        for k in ordered_metrics:
            val = r.get("latest_metrics", {}).get(k)
            row[k] = val if val is not None else "-"
        table_rows.append(row)

    lines.append("")
    lines.append(metric_table(table_rows, columns, title="📋 Run Comparison"))

    # ── Hyperparameter diff ──
    all_hp_keys: list[str] = []
    for r in runs:
        for k in r.get("hyperparams", {}):
            if k not in all_hp_keys:
                all_hp_keys.append(k)

    if all_hp_keys:
        lines.append("")
        lines.append("⚙️  Hyperparameter Diff")
        lines.append(DASH_LINE * 56)

        # Find diffs only
        for k in all_hp_keys:
            vals = [r.get("hyperparams", {}).get(k, "-") for r in runs]
            # Show if values differ
            if len(set(str(v) for v in vals)) > 1:
                lines.append(f"  {k}:")
                for i, r in enumerate(runs):
                    rname = r.get("name", r.get("id", "?"))[:20]
                    v = r.get("hyperparams", {}).get(k, "-")
                    lines.append(f"    {MARKERS[i % len(MARKERS)]} {rname}: {v}")
            else:
                lines.append(f"  {k}: {vals[0]}  (same)")

    # ── Overlay line chart (loss curves) ──
    overlay_series: dict[str, list[float]] = {}
    max_points = 0
    for r in runs:
        mh = r.get("metrics_history", [])
        loss_vals = [e["loss"] for e in mh if "loss" in e]
        if len(loss_vals) >= 2:
            rname = r.get("name", r.get("id", "?"))[:18]
            overlay_series[rname] = loss_vals
            max_points = max(max_points, len(loss_vals))

    if overlay_series:
        x_labels = list(range(1, max_points + 1))
        lines.append("")
        lines.append(line_chart(
            x_labels,
            overlay_series,
            title="Loss Curves Overlay",
            height=chart_height,
            width=chart_width,
        ))

    # ── Success rate comparison bar chart ──
    sr_labels = []
    sr_values = []
    for r in runs:
        sr = r.get("latest_metrics", {}).get("success_rate")
        if sr is not None:
            sr_labels.append(r.get("name", r.get("id", "?"))[:20])
            sr_values.append(sr)

    if sr_labels:
        lines.append("")
        lines.append(bar_chart(sr_labels, sr_values, title="Success Rate Comparison", max_bar_width=25))

    return "\n".join(lines)
