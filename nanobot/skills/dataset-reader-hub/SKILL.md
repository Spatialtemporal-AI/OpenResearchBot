---
name: dataset-reader-hub
description: Read, visualize, and analyze robotics datasets with reusable dataset readers. Use when a user asks to load/inspect/visualize/analyze datasets like RoboMIND, RoboChallenge, LeRobot, AgiBot, GM100, RoboCoin, or when a new dataset needs a reader based on existing templates.
---

# Dataset Reader Hub

## Preconditions
- Activate the conda environment before running any reader: `conda activate a1`.
- Check each reader file header for required env vars (for example, `NORM_STATS_PATH` in `robomind_datasets.py`).
 - Optional helper: `scripts/run_in_a1.sh <command...>` runs a command inside `conda activate a1`.

## Token safety (required)
- Never use `read_file` on large data files such as:
  - `**/states/*.jsonl`, `**/states/*.json`, `**/*.mp4`
- If user asks for schema/format/shape overview, always use the reader's `--mode inspect` first:
  - RoboChallenge: `rc_reader.py --mode inspect`
  - RoboMIND: `robomind_datasets.py --mode inspect`
  - RoboCOIN: `robocoin_reader.py --mode inspect`
- `inspect` mode must only read lightweight metadata and tiny samples (never full trajectory/video payloads).

## Reader templates
Location: `nanobot/skills/dataset-reader-hub/assets/reader_templates/`

- `robomind_datasets.py` - RoboMIND reader; read/stats/visualize/build_index/inspect modes
- `rc_reader.py` - RoboChallenge reader; read/stats/visualize/inspect modes
- `lerobot_datasets.py` - LeRobot dataset wrapper; read/transform helpers
- `agibot_dataset.py`, `agibot_datasets.py` - AgiBot dataset readers
- `gm100_reader.py` - GM100 reader (LeRobot-style)
- `robocoin_reader.py` - RoboCoin reader; read/stats/inspect modes

## Common commands
Use `scripts/run_in_a1.sh` to avoid forgetting `conda activate a1`.

RoboChallenge (visualize a few episodes):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/rc_reader.py \
  --mode visualize \
  --dataset_path /path/to/RoboChallenge \
  --env_name turn_on_faucet \
  --embodiment ALOHA \
  --random_episodes 4 \
  --output_dir outputs/rc_videos
```

RoboChallenge (stats):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/rc_reader.py \
  --mode stats \
  --dataset_path /path/to/RoboChallenge \
  --env_name turn_on_faucet \
  --embodiment ALOHA \
  --output_path outputs/rc_stats.json
```

RoboChallenge (safe format/shape inspect, preferred for "看数据格式"):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/rc_reader.py \
  --mode inspect \
  --dataset_path /path/to/rc_task_or_root \
  --env_name set_the_plates \
  --inspect_max_tasks 1 \
  --inspect_max_episodes 3 \
  --inspect_max_state_lines 2
```

RoboMIND (read one frame):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/robomind_datasets.py \
  --mode read \
  --embodiment h5_agilex_3rgb \
  --dataset_path /path/to/robomind \
  --env_name some_task \
  --episode_idx 0 \
  --frame_idx 0
```

RoboMIND (safe format/shape inspect):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/robomind_datasets.py \
  --mode inspect \
  --dataset_path /path/to/robomind \
  --embodiment h5_agilex_3rgb \
  --inspect_max_envs 2 \
  --inspect_max_episodes 2
```

RoboMIND (visualize):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/robomind_datasets.py \
  --mode visualize \
  --embodiment h5_agilex_3rgb \
  --dataset_path /path/to/robomind \
  --env_name some_task \
  --output_dir outputs/robomind_videos \
  --num_episodes 4
```

LeRobot (quick test / stats / visualize):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/lerobot_datasets.py \
  --mode test \
  --dataset_path /path/to/lerobot_dataset
```

RoboCOIN (safe format/shape inspect):
```bash
nanobot/skills/dataset-reader-hub/scripts/run_in_a1.sh \
  python nanobot/skills/dataset-reader-hub/assets/reader_templates/robocoin_reader.py \
  --mode inspect \
  --dataset_path /path/to/robocoin_root_or_task \
  --inspect_max_tasks 2 \
  --inspect_max_jsonl_lines 2
```

## Common outputs
Prefer returning a dict aligned with RoboMIND/LeRobot conventions when reading:
`question`, `action`, `proprio`, `images`, `metadata` (plus `timestep`/`style`/`answer` if present).
Match the exact keys used in the selected template.

## Handle a user request
1. Identify dataset name, path/URL, and task (read / visualize / analyze).
2. If user asks for schema/format/shape overview, run the matching reader's `--mode inspect` first (token-safe path).
3. Pick the closest template and open it for usage examples (`if __name__ == "__main__"` blocks often show CLI).
4. Run the reader or import its class. If visualization/stats are requested, use the reader's `--mode` or helper functions.
5. If the selected reader lacks visualization or stats, add them by borrowing the patterns from RoboMIND/RoboChallenge templates.
6. Summarize outputs and return paths to generated files (videos, stats JSON).

## Create a new reader (unseen dataset)
1. Inspect dataset structure (file formats, sensors, action/state fields, video layout).
2. Choose the closest template:
   - HDF5 trajectory data -> `robomind_datasets.py`
   - LeRobot-format datasets -> `lerobot_datasets.py` or `gm100_reader.py`
   - JSONL + MP4 tasks -> `robochallenge_reader.py`
3. Implement a `Dataset`-style reader with:
   - `read` mode returning the standard dict
   - `stats` mode with min/max/mean/std/q01/q99 (reuse RoboMIND/RoboChallenge patterns)
   - `visualize` mode to export MP4s (borrow the video export logic from RoboMIND/RoboChallenge)
4. Add a CLI (`argparse`) with `--mode`, `--dataset_path`, and other dataset-specific flags.
5. Update the template list above if a new reader is added.
