---
name: dataset-reader-hub
description: Read, visualize, and analyze robotics datasets with reusable dataset readers. Use when a user asks to load/inspect/visualize/analyze datasets like RoboMIND, RoboChallenge, LeRobot, AgiBot, GM100, RoboCoin, or when a new dataset needs a reader based on existing templates.
---

# Dataset Reader Hub

## Preconditions
- Activate the conda environment before running any reader: `conda activate a1`.
- Check each reader file header for required env vars (for example, `NORM_STATS_PATH` in `robomind_datasets.py`).
 - Optional helper: `scripts/run_in_a1.sh <command...>` runs a command inside `conda activate a1`.

## Reader templates
Location: `nanobot/skills/dataset-reader-hub/assets/reader_templates/`

- `robomind_datasets.py` - RoboMIND reader; read/stats/visualize modes
- `rc_reader.py` - RoboChallenge reader; read/stats/visualize modes
- `lerobot_datasets.py` - LeRobot dataset wrapper; read/transform helpers
- `agibot_dataset.py`, `agibot_datasets.py` - AgiBot dataset readers
- `gm100_reader.py` - GM100 reader (LeRobot-style)
- `robocoin_reader.py` - RoboCoin reader

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

## Common outputs
Prefer returning a dict aligned with RoboMIND/LeRobot conventions when reading:
`question`, `action`, `proprio`, `images`, `metadata` (plus `timestep`/`style`/`answer` if present).
Match the exact keys used in the selected template.

## Handle a user request
1. Identify dataset name, path/URL, and task (read / visualize / analyze).
2. Pick the closest template and open it for usage examples (`if __name__ == "__main__"` blocks often show CLI).
3. Run the reader or import its class. If visualization/stats are requested, use the reader's `--mode` or helper functions.
4. If the selected reader lacks visualization or stats, add them by borrowing the patterns from RoboMIND/RoboChallenge templates.
5. Summarize outputs and return paths to generated files (videos, stats JSON).

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
