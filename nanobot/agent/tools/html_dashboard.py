"""HTML dashboard generator for task_tracker & training_tracker.

Generates a self-contained HTML file with interactive charts (Chart.js CDN),
responsive design, and modern styling. No server needed — just open in browser.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def generate_dashboard(workspace: Path) -> Path:
    """Generate a full HTML dashboard and return the file path."""
    tasks_data = _load_json(workspace / "research" / "tasks.json")
    training_data = _load_json(workspace / "research" / "training_runs.json")

    tasks = tasks_data.get("tasks", [])
    runs = training_data.get("runs", [])

    html = _build_html(tasks, runs)

    out_path = workspace / "research" / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_training_dashboard(workspace: Path, run_id: str = "", run_ids: list[str] | None = None) -> Path:
    """Generate an HTML dashboard for training runs only."""
    training_data = _load_json(workspace / "research" / "training_runs.json")
    runs = training_data.get("runs", [])

    if run_ids and len(run_ids) >= 2:
        selected = [r for r in runs if r["id"] in run_ids]
    elif run_id:
        selected = [r for r in runs if r["id"] == run_id]
    else:
        selected = runs

    html = _build_html([], selected)
    out_path = workspace / "research" / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_task_dashboard(workspace: Path) -> Path:
    """Generate an HTML dashboard for tasks only."""
    tasks_data = _load_json(workspace / "research" / "tasks.json")
    tasks = tasks_data.get("tasks", [])

    html = _build_html(tasks, [])
    out_path = workspace / "research" / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ─── HTML Builder ────────────────────────────────────────────────

def _build_html(tasks: list[dict], runs: list[dict], *, live_mode: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare JSON data for embedding
    tasks_json = json.dumps(tasks, ensure_ascii=False)
    runs_json = json.dumps(runs, ensure_ascii=False)

    live_flag_js = "window.__LIVE__ = true;" if live_mode else ""
    live_indicator = (
        ' <span style="display:inline-block;background:#ef4444;color:#fff;'
        "font-size:0.7rem;padding:2px 10px;border-radius:12px;margin-left:12px;"
        'animation:pulse 2s infinite;">● LIVE</span>'
    ) if live_mode else ""
    live_ts = (
        '<p class="subtitle" id="live-ts" style="color:var(--accent2);">'
        "Connected — auto-refreshing every 3s</p>"
    ) if live_mode else ""
    subtitle = "Live Dashboard" if live_mode else f"Generated at {now}"
    live_css = (
        "\n@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }"
    ) if live_mode else ""
    live_js = f"\n{_LIVE_JS}" if live_mode else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔬 Research Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
{_CSS}{live_css}
</style>
</head>
<body>

<header>
  <h1>🔬 Research Dashboard{live_indicator}</h1>
  <p class="subtitle">{subtitle}</p>
  {live_ts}
</header>

<main id="app">
  <div id="task-section"></div>
  <div id="training-section"></div>
</main>

<script>
{live_flag_js}
const TASKS = {tasks_json};
const RUNS = {runs_json};

{_JS}{live_js}
</script>

</body>
</html>"""


def generate_live_html(workspace: Path) -> str:
    """Generate dashboard HTML string with live-refresh enabled."""
    tasks_data = _load_json(workspace / "research" / "tasks.json")
    runs_data = _load_json(workspace / "research" / "training_runs.json")
    return _build_html(
        tasks_data.get("tasks", []),
        runs_data.get("runs", []),
        live_mode=True,
    )


def load_tracker_data(workspace: Path) -> dict:
    """Load current tasks and runs data as a dict."""
    tasks_data = _load_json(workspace / "research" / "tasks.json")
    runs_data = _load_json(workspace / "research" / "training_runs.json")
    return {
        "tasks": tasks_data.get("tasks", []),
        "runs": runs_data.get("runs", []),
    }


# ─── CSS ─────────────────────────────────────────────────────────

_CSS = """\
:root {
  --bg: #0f1117;
  --surface: #1a1d2e;
  --surface2: #232640;
  --border: #2d3154;
  --text: #e2e4f0;
  --text-dim: #8b8fa8;
  --accent: #6c7bff;
  --accent2: #00d4aa;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
  --info: #3b82f6;
  --radius: 12px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

header {
  background: linear-gradient(135deg, #1a1d2e 0%, #232640 100%);
  border-bottom: 1px solid var(--border);
  padding: 24px 32px;
  text-align: center;
}
header h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
.subtitle { color: var(--text-dim); font-size: 0.85rem; margin-top: 4px; }

main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px 16px;
}

/* Cards */
.section-title {
  font-size: 1.3rem;
  font-weight: 600;
  margin: 32px 0 16px;
  padding-left: 12px;
  border-left: 4px solid var(--accent);
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  transition: transform 0.15s, box-shadow 0.15s;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.stat-card {
  text-align: center;
}
.stat-card .stat-value {
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--accent);
}
.stat-card .stat-label {
  font-size: 0.85rem;
  color: var(--text-dim);
  margin-top: 4px;
}

.chart-card {
  padding: 20px;
}
.chart-card h3 {
  font-size: 0.95rem;
  font-weight: 600;
  margin-bottom: 12px;
  color: var(--text-dim);
}

.wide-card {
  grid-column: 1 / -1;
}

/* Tables */
.table-wrap {
  overflow-x: auto;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
th {
  background: var(--surface2);
  font-weight: 600;
  text-align: left;
  padding: 12px 16px;
  white-space: nowrap;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
}
td {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--surface2); }

/* Badges */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.badge-doing    { background: #1e3a5f; color: #60a5fa; }
.badge-todo     { background: #2d2d3d; color: #a1a5c0; }
.badge-done     { background: #14532d; color: #4ade80; }
.badge-blocked  { background: #450a0a; color: #f87171; }
.badge-running  { background: #1e3a5f; color: #60a5fa; }
.badge-queued   { background: #3b2e1a; color: #fbbf24; }
.badge-completed{ background: #14532d; color: #4ade80; }
.badge-failed   { background: #450a0a; color: #f87171; }
.badge-stopped  { background: #2d2d3d; color: #a1a5c0; }

.badge-high   { background: #450a0a; color: #f87171; }
.badge-medium { background: #3b2e1a; color: #fbbf24; }
.badge-low    { background: #1a2e1a; color: #86efac; }

.tag {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 4px;
  font-size: 0.72rem;
  background: var(--surface2);
  color: var(--text-dim);
  margin-right: 4px;
}

/* Progress bar */
.progress-wrap { margin: 8px 0; }
.progress-bar {
  height: 8px;
  background: var(--surface2);
  border-radius: 4px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

/* Run detail panels */
.run-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 20px;
  overflow: hidden;
}
.run-panel-header {
  padding: 16px 20px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.run-panel-header h3 { font-size: 1.05rem; }
.run-panel-body { padding: 20px; }
.run-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.run-meta-item {
  font-size: 0.85rem;
}
.run-meta-item .label {
  color: var(--text-dim);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.notes-list {
  list-style: none;
  font-size: 0.85rem;
}
.notes-list li {
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  color: var(--text-dim);
}
.notes-list li:last-child { border-bottom: none; }
.notes-list .note-time { color: var(--accent); font-size: 0.75rem; margin-right: 8px; }

.empty-state {
  text-align: center;
  color: var(--text-dim);
  padding: 48px 16px;
  font-size: 0.95rem;
}

/* Responsive */
@media (max-width: 640px) {
  header { padding: 16px; }
  header h1 { font-size: 1.3rem; }
  main { padding: 12px 8px; }
  .card { padding: 14px; }
  .stat-card .stat-value { font-size: 1.8rem; }
  td, th { padding: 8px 10px; font-size: 0.8rem; }
}
"""


# ─── JavaScript ──────────────────────────────────────────────────

_JS = """\
// ─── Color palette ───
const COLORS = {
  accent:  '#6c7bff',
  accent2: '#00d4aa',
  success: '#22c55e',
  warning: '#f59e0b',
  danger:  '#ef4444',
  info:    '#3b82f6',
  purple:  '#a855f7',
  pink:    '#ec4899',
};
const CHART_COLORS = ['#6c7bff','#00d4aa','#f59e0b','#ef4444','#a855f7','#ec4899','#3b82f6','#22c55e'];

Chart.defaults.color = '#8b8fa8';
Chart.defaults.borderColor = '#2d3154';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

// ─── Utilities ───
function badge(text, cls) {
  return `<span class="badge badge-${cls || text}">${text}</span>`;
}
function tag(text) {
  return `<span class="tag">${text}</span>`;
}

// ═══════════════════════════════════════════
//  TASK SECTION
// ═══════════════════════════════════════════

function renderTasks(tasks) {
  const section = document.getElementById('task-section');
  if (!tasks.length) {
    section.innerHTML = '';
    return;
  }

  const byStatus = {};
  const byPriority = {};
  tasks.forEach(t => {
    byStatus[t.status] = (byStatus[t.status] || 0) + 1;
    if (t.status !== 'done') {
      byPriority[t.priority || 'medium'] = (byPriority[t.priority || 'medium'] || 0) + 1;
    }
  });

  const doneCount = byStatus['done'] || 0;
  const total = tasks.length;
  const pct = total ? Math.round(doneCount / total * 100) : 0;

  let html = '<h2 class="section-title">📋 Research Tasks</h2>';

  // ── Stat cards ──
  html += '<div class="card-grid">';
  html += `<div class="card stat-card"><div class="stat-value">${total}</div><div class="stat-label">Total Tasks</div></div>`;
  html += `<div class="card stat-card"><div class="stat-value" style="color:${COLORS.info}">${byStatus['doing']||0}</div><div class="stat-label">In Progress</div></div>`;
  html += `<div class="card stat-card"><div class="stat-value" style="color:${COLORS.success}">${doneCount}</div><div class="stat-label">Completed</div></div>`;
  html += `<div class="card stat-card"><div class="stat-value" style="color:${COLORS.danger}">${byStatus['blocked']||0}</div><div class="stat-label">Blocked</div></div>`;
  html += '</div>';

  // ── Progress ──
  const progressColor = pct >= 80 ? COLORS.success : pct >= 40 ? COLORS.warning : COLORS.accent;
  html += `<div class="card" style="margin-bottom:24px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
      <span style="font-size:0.9rem;">Overall Progress</span>
      <span style="font-weight:600;">${pct}% (${doneCount}/${total})</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${progressColor};"></div></div>
  </div>`;

  // ── Charts row ──
  html += '<div class="card-grid">';
  html += '<div class="card chart-card"><h3>Status Distribution</h3><canvas id="taskStatusChart"></canvas></div>';
  html += '<div class="card chart-card"><h3>Priority (Active)</h3><canvas id="taskPriorityChart"></canvas></div>';
  html += '</div>';

  // ── Table ──
  html += '<div class="card wide-card" style="padding:0;">';
  html += '<div style="padding:16px 20px;border-bottom:1px solid var(--border);"><h3 style="font-size:0.95rem;">All Tasks</h3></div>';
  html += '<div class="table-wrap"><table>';
  html += '<tr><th>ID</th><th>Title</th><th>Status</th><th>Priority</th><th>Tags</th><th>Updated</th></tr>';
  // Sort: doing > blocked > todo > done
  const statusOrder = {doing:0, blocked:1, todo:2, done:3};
  const sorted = [...tasks].sort((a,b) => (statusOrder[a.status]??9) - (statusOrder[b.status]??9));
  sorted.forEach(t => {
    const tags = (t.tags || []).map(x => tag(x)).join(' ');
    const updated = t.updated ? t.updated.replace('T',' ') : '-';
    html += `<tr>
      <td style="font-family:monospace;font-size:0.8rem;color:var(--text-dim);">${t.id}</td>
      <td style="font-weight:500;white-space:normal;min-width:200px;">${t.title}</td>
      <td>${badge(t.status)}</td>
      <td>${badge(t.priority || 'medium', t.priority || 'medium')}</td>
      <td>${tags}</td>
      <td style="color:var(--text-dim);font-size:0.8rem;">${updated}</td>
    </tr>`;
  });
  html += '</table></div></div>';

  section.innerHTML = html;

  // ── Draw charts ──
  const statusLabels = ['doing','todo','blocked','done'];
  const statusColors = [COLORS.info, '#6b7280', COLORS.danger, COLORS.success];
  new Chart(document.getElementById('taskStatusChart'), {
    type: 'doughnut',
    data: {
      labels: statusLabels.filter(s => byStatus[s]),
      datasets: [{
        data: statusLabels.filter(s => byStatus[s]).map(s => byStatus[s]),
        backgroundColor: statusLabels.filter(s => byStatus[s]).map((s,i) => statusColors[statusLabels.indexOf(s)]),
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      cutout: '65%',
      plugins: { legend: { position: 'bottom', labels: { padding: 12 } } },
    }
  });

  const prioLabels = ['high','medium','low'];
  const prioColors = [COLORS.danger, COLORS.warning, COLORS.success];
  new Chart(document.getElementById('taskPriorityChart'), {
    type: 'doughnut',
    data: {
      labels: prioLabels.filter(p => byPriority[p]),
      datasets: [{
        data: prioLabels.filter(p => byPriority[p]).map(p => byPriority[p]),
        backgroundColor: prioLabels.filter(p => byPriority[p]).map(p => prioColors[prioLabels.indexOf(p)]),
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      cutout: '65%',
      plugins: { legend: { position: 'bottom', labels: { padding: 12 } } },
    }
  });
}

// ═══════════════════════════════════════════
//  TRAINING SECTION
// ═══════════════════════════════════════════

function renderTraining(runs) {
  const section = document.getElementById('training-section');
  if (!runs.length) {
    section.innerHTML = '';
    return;
  }

  let html = '<h2 class="section-title">🏋️ Training Runs</h2>';

  // ── Overview stats ──
  const byStatus = {};
  const vlaCount = runs.filter(r => r.vla_config && Object.keys(r.vla_config).length).length;
  runs.forEach(r => { byStatus[r.status] = (byStatus[r.status]||0) + 1; });

  html += '<div class="card-grid">';
  html += `<div class="card stat-card"><div class="stat-value">${runs.length}</div><div class="stat-label">Total Runs</div></div>`;
  html += `<div class="card stat-card"><div class="stat-value" style="color:${COLORS.info}">${byStatus['running']||0}</div><div class="stat-label">Running</div></div>`;
  html += `<div class="card stat-card"><div class="stat-value" style="color:${COLORS.success}">${byStatus['completed']||0}</div><div class="stat-label">Completed</div></div>`;
  html += `<div class="card stat-card"><div class="stat-value" style="color:${COLORS.purple}">${vlaCount}</div><div class="stat-label">VLA Runs</div></div>`;
  html += '</div>';

  // ── Overview table ──
  if (runs.length > 1) {
    html += '<div class="card wide-card" style="padding:0;margin-bottom:24px;">';
    html += '<div style="padding:16px 20px;border-bottom:1px solid var(--border);"><h3 style="font-size:0.95rem;">All Runs</h3></div>';
    html += '<div class="table-wrap"><table>';
    html += '<tr><th>ID</th><th>Name</th><th>Model</th><th>Status</th><th>Loss</th><th>Success Rate</th><th>GPU</th></tr>';
    runs.forEach(r => {
      const lm = r.latest_metrics || {};
      html += `<tr>
        <td style="font-family:monospace;font-size:0.8rem;color:var(--text-dim);">${r.id}</td>
        <td style="font-weight:500;">${r.name || '-'}</td>
        <td>${r.model || '-'}</td>
        <td>${badge(r.status)}</td>
        <td>${lm.loss != null ? lm.loss : '-'}</td>
        <td>${lm.success_rate != null ? lm.success_rate : '-'}</td>
        <td style="font-size:0.8rem;color:var(--text-dim);">${r.gpu_info || '-'}</td>
      </tr>`;
    });
    html += '</table></div></div>';
  }

  // ── Comparison charts (if 2+ runs with metrics) ──
  const runsWithMetrics = runs.filter(r => (r.metrics_history||[]).length >= 2);
  if (runsWithMetrics.length >= 2) {
    html += '<div class="card-grid">';
    html += '<div class="card chart-card"><h3>📉 Loss Curves Comparison</h3><canvas id="lossCompareChart"></canvas></div>';

    // Success rate bar chart
    const runsWithSR = runs.filter(r => r.latest_metrics && r.latest_metrics.success_rate != null);
    if (runsWithSR.length >= 2) {
      html += '<div class="card chart-card"><h3>🎯 Success Rate Comparison</h3><canvas id="srCompareChart"></canvas></div>';
    }
    html += '</div>';
  }

  // ── Per-run detail panels ──
  runs.forEach((run, idx) => {
    const mh = run.metrics_history || [];
    const hp = run.hyperparams || {};
    const vla = run.vla_config || {};
    const isVLA = Object.keys(vla).length > 0;
    const lm = run.latest_metrics || {};
    const totalEpochs = hp.epochs || 0;
    const lastEpoch = lm.epoch || 0;
    const pct = totalEpochs ? Math.round(lastEpoch / totalEpochs * 100) : 0;

    html += `<div class="run-panel">`;
    html += `<div class="run-panel-header">
      <h3>${isVLA ? '🤖 ' : ''}${run.name || 'Unnamed Run'}</h3>
      <div>${badge(run.status)} <span style="font-size:0.8rem;color:var(--text-dim);margin-left:8px;">${run.id}</span></div>
    </div>`;
    html += '<div class="run-panel-body">';

    // Meta
    html += '<div class="run-meta">';
    html += `<div class="run-meta-item"><div class="label">Model</div><div>${run.model||'-'}</div></div>`;
    html += `<div class="run-meta-item"><div class="label">Dataset</div><div>${run.dataset||'-'}</div></div>`;
    html += `<div class="run-meta-item"><div class="label">GPU</div><div>${run.gpu_info||'-'}</div></div>`;
    html += `<div class="run-meta-item"><div class="label">Created</div><div>${(run.created||'').replace('T',' ')}</div></div>`;
    html += '</div>';

    // Progress bar
    if (totalEpochs) {
      const pc = pct >= 80 ? COLORS.success : pct >= 40 ? COLORS.warning : COLORS.accent;
      html += `<div style="margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:0.85rem;">Epoch Progress</span>
          <span style="font-weight:600;font-size:0.85rem;">${lastEpoch} / ${totalEpochs} (${pct}%)</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${pc};"></div></div>
      </div>`;
    }

    // Charts for this run
    if (mh.length >= 2) {
      html += '<div class="card-grid" style="margin-bottom:16px;">';
      html += `<div class="card chart-card" style="background:var(--surface2);"><h3>Training Curves</h3><canvas id="runChart${idx}"></canvas></div>`;
      // Latest metrics card
      if (Object.keys(lm).length) {
        html += '<div class="card" style="background:var(--surface2);">';
        html += '<h3 style="font-size:0.95rem;margin-bottom:12px;color:var(--text-dim);">Latest Metrics</h3>';
        for (const [k,v] of Object.entries(lm)) {
          html += `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:0.88rem;">
            <span style="color:var(--text-dim);">${k}</span>
            <span style="font-weight:600;">${v}</span>
          </div>`;
        }
        html += '</div>';
      }
      html += '</div>';
    }

    // Hyperparameters + VLA config
    const hasHP = Object.keys(hp).length > 0;
    const hasVLA = Object.keys(vla).length > 0;
    if (hasHP || hasVLA) {
      html += '<div class="card-grid" style="margin-bottom:16px;">';
      if (hasHP) {
        html += '<div class="card" style="background:var(--surface2);">';
        html += '<h3 style="font-size:0.95rem;margin-bottom:10px;color:var(--text-dim);">⚙️ Hyperparameters</h3>';
        for (const [k,v] of Object.entries(hp)) {
          html += `<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.85rem;">
            <span style="color:var(--text-dim);">${k}</span><span>${v}</span>
          </div>`;
        }
        html += '</div>';
      }
      if (hasVLA) {
        html += '<div class="card" style="background:var(--surface2);">';
        html += '<h3 style="font-size:0.95rem;margin-bottom:10px;color:var(--text-dim);">🤖 VLA Configuration</h3>';
        const vlaLabels = {action_space:'Action Space',observation_space:'Observation Space',
          embodiment:'Embodiment',environment:'Environment',task_suite:'Task Suite',
          action_tokenizer:'Tokenizer',backbone:'Backbone'};
        for (const [k,v] of Object.entries(vla)) {
          html += `<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.85rem;gap:12px;">
            <span style="color:var(--text-dim);white-space:nowrap;">${vlaLabels[k]||k}</span><span style="text-align:right;">${v}</span>
          </div>`;
        }
        html += '</div>';
      }
      html += '</div>';
    }

    // Metrics history table
    if (mh.length) {
      const allKeys = [];
      mh.forEach(e => { Object.keys(e).forEach(k => { if (k !== 'time' && !allKeys.includes(k)) allKeys.push(k); }); });
      const preferred = ['epoch','step','loss','eval_loss','action_mse','success_rate'];
      const orderedKeys = [...preferred.filter(k => allKeys.includes(k)), ...allKeys.filter(k => !preferred.includes(k))];

      html += '<div style="margin-bottom:16px;">';
      html += '<h3 style="font-size:0.95rem;margin-bottom:8px;color:var(--text-dim);">📋 Metrics History</h3>';
      html += '<div class="table-wrap"><table>';
      html += '<tr>' + orderedKeys.map(k => `<th>${k}</th>`).join('') + '</tr>';
      mh.forEach(e => {
        html += '<tr>' + orderedKeys.map(k => `<td>${e[k] != null ? e[k] : '-'}</td>`).join('') + '</tr>';
      });
      html += '</table></div></div>';
    }

    // Checkpoints
    const cps = run.checkpoints || [];
    if (cps.length) {
      html += `<h3 style="font-size:0.95rem;margin-bottom:8px;color:var(--text-dim);">💾 Checkpoints (${cps.length})</h3>`;
      html += '<ul class="notes-list" style="margin-bottom:16px;">';
      cps.forEach(cp => { html += `<li style="font-family:monospace;font-size:0.82rem;">📦 ${cp}</li>`; });
      html += '</ul>';
    }

    // Notes
    const notes = run.notes || [];
    if (notes.length) {
      html += `<h3 style="font-size:0.95rem;margin-bottom:8px;color:var(--text-dim);">📝 Notes (${notes.length})</h3>`;
      html += '<ul class="notes-list">';
      notes.forEach(n => {
        html += `<li><span class="note-time">${(n.time||'').replace('T',' ')}</span>${n.content||''}</li>`;
      });
      html += '</ul>';
    }

    html += '</div></div>'; // panel-body, run-panel
  });

  section.innerHTML = html;

  // ── Draw per-run charts ──
  runs.forEach((run, idx) => {
    const canvas = document.getElementById(`runChart${idx}`);
    if (!canvas) return;
    const mh = run.metrics_history || [];

    const allKeys = [];
    mh.forEach(e => { Object.keys(e).forEach(k => { if (k !== 'time' && k !== 'epoch' && k !== 'step' && !allKeys.includes(k)) allKeys.push(k); }); });

    const xLabels = mh.map(e => e.epoch != null ? 'E' + e.epoch : (e.step != null ? 'S' + e.step : ''));
    const datasets = allKeys.map((key, i) => ({
      label: key,
      data: mh.map(e => e[key] != null ? e[key] : null),
      borderColor: CHART_COLORS[i % CHART_COLORS.length],
      backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '20',
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.3,
      spanGaps: true,
    }));

    new Chart(canvas, {
      type: 'line',
      data: { labels: xLabels, datasets },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { padding: 12 } } },
        scales: {
          y: { grid: { color: '#2d315440' } },
          x: { grid: { color: '#2d315420' } },
        },
      }
    });
  });

  // ── Loss comparison chart ──
  const lossCanvas = document.getElementById('lossCompareChart');
  if (lossCanvas) {
    const rwm = runs.filter(r => (r.metrics_history||[]).length >= 2);
    const datasets = rwm.map((r, i) => {
      const mh = r.metrics_history || [];
      return {
        label: r.name || r.id,
        data: mh.map(e => ({x: e.epoch || e.step || 0, y: e.loss})).filter(d => d.y != null),
        borderColor: CHART_COLORS[i % CHART_COLORS.length],
        borderWidth: 2,
        pointRadius: 4,
        tension: 0.3,
      };
    });
    new Chart(lossCanvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { padding: 12 } } },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'Epoch' }, grid: { color: '#2d315420' } },
          y: { title: { display: true, text: 'Loss' }, grid: { color: '#2d315440' } },
        },
      }
    });
  }

  // ── Success rate comparison ──
  const srCanvas = document.getElementById('srCompareChart');
  if (srCanvas) {
    const rwsr = runs.filter(r => r.latest_metrics && r.latest_metrics.success_rate != null);
    new Chart(srCanvas, {
      type: 'bar',
      data: {
        labels: rwsr.map(r => r.name || r.id),
        datasets: [{
          label: 'Success Rate',
          data: rwsr.map(r => r.latest_metrics.success_rate),
          backgroundColor: rwsr.map((_, i) => CHART_COLORS[i % CHART_COLORS.length] + 'cc'),
          borderColor: rwsr.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
          borderWidth: 1,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, max: 1.0, grid: { color: '#2d315440' },
               title: { display: true, text: 'Success Rate' } },
          x: { grid: { display: false } },
        },
      }
    });
  }
}

// ─── Init ───
renderTasks(TASKS);
renderTraining(RUNS);

if (!TASKS.length && !RUNS.length) {
  document.getElementById('app').innerHTML = '<div class="empty-state">📭 No data yet. Use the tracker tools to create tasks and training runs.</div>';
}
"""


# ─── Live-mode JavaScript (AJAX polling) ─────────────────────────

_LIVE_JS = """\
// ─── Live Auto-Refresh ───
if (window.__LIVE__) {
  const _tsEl = document.getElementById('live-ts');

  async function _refreshData() {
    try {
      const resp = await fetch('/api/data');
      const data = await resp.json();

      // Destroy all existing Chart.js instances before re-rendering
      document.querySelectorAll('canvas').forEach(canvas => {
        const chart = Chart.getChart(canvas);
        if (chart) chart.destroy();
      });

      // Re-render sections with fresh data
      renderTasks(data.tasks || []);
      renderTraining(data.runs || []);

      if (!(data.tasks || []).length && !(data.runs || []).length) {
        document.getElementById('app').innerHTML =
          '<div class="empty-state">📭 No data yet. Use tracker API or Agent to add tasks and training runs.</div>';
      }

      if (_tsEl) _tsEl.textContent = 'Last updated: ' + new Date().toLocaleTimeString();
    } catch (e) {
      if (_tsEl) _tsEl.textContent = '⚠️ Connection lost — retrying...';
    }
  }

  // Poll every 3 seconds
  setInterval(_refreshData, 3000);
}
"""
