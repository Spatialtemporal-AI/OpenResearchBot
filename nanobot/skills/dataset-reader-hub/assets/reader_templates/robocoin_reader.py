"""
RoboCOIN dataset reader.

This reader wraps LeRobotDataset (RoboCOIN is LeRobot v2.1 compatible) and
returns the same dict structure as `rc_reader.py` for joint training.

Output dict keys:
  - question, timestep, answer, style, action, action_pad_mask, proprio, images, metadata
"""

from __future__ import annotations

import argparse
import json
import os
import re
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from olmo.data.dataset import Dataset
except Exception:
    # Allow inspect mode to run without full training environment.
    class Dataset:  # type: ignore[override]
        pass

try:
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
        CODEBASE_VERSION,
    )
    from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
except Exception as exc:  # pragma: no cover - handled at runtime if lerobot missing
    LeRobotDataset = None
    LeRobotDatasetMetadata = None
    CODEBASE_VERSION = None
    ImageTransforms = None
    ImageTransformsConfig = None
    _LEROBOT_IMPORT_ERROR = exc
else:
    _LEROBOT_IMPORT_ERROR = None


def _build_image_transforms_cfg():
    if ImageTransformsConfig is None:
        return None
    defaults = {
        "normalize": True,
        "resize": None,
        "crop": None,
        "random_crop": False,
        "random_horizontal_flip": False,
        "random_vertical_flip": False,
    }
    try:
        sig = inspect.signature(ImageTransformsConfig)
        filtered = {k: v for k, v in defaults.items() if k in sig.parameters}
        return ImageTransformsConfig(**filtered)
    except Exception:
        try:
            return ImageTransformsConfig()
        except Exception:
            return None


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _format_image(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr.squeeze(0)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.dtype != np.uint8:
        if arr.size > 0 and np.nanmax(arr) > 1.5:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
    return arr


def _to_scalar(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    arr = _to_numpy(x)
    if np.isscalar(arr):
        return float(arr)
    if isinstance(arr, np.ndarray) and arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def _inspect_shape_of_list(value: Any) -> List[int]:
    dims: List[int] = []
    cur = value
    while isinstance(cur, list):
        dims.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return dims


def _inspect_leaf_type(value: Any) -> str:
    cur = value
    while isinstance(cur, list) and cur:
        cur = cur[0]
    if isinstance(cur, list):
        return "list"
    return type(cur).__name__


def _inspect_value_signature(value: Any) -> str:
    if isinstance(value, list):
        dims = _inspect_shape_of_list(value)
        dims_str = "x".join(str(d) for d in dims) if dims else "empty"
        return f"list[{dims_str}]<{_inspect_leaf_type(value)}>"
    if isinstance(value, dict):
        return f"dict<{len(value)} keys>"
    return f"scalar<{type(value).__name__}>"


def _inspect_jsonl(path: Path, max_lines: int) -> Dict[str, Any]:
    key_signatures: Dict[str, set] = {}
    sampled_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if sampled_lines >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sampled_lines += 1
            if not isinstance(payload, dict):
                continue
            for k, v in payload.items():
                key_signatures.setdefault(k, set()).add(_inspect_value_signature(v))
    return {
        "file": str(path),
        "sampled_lines": sampled_lines,
        "keys": {k: sorted(list(v)) for k, v in sorted(key_signatures.items())},
    }


def _discover_robocoin_task_dirs(root: Path, env_names: Optional[List[str]] = None) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    # Single task folder
    if (root / "meta").exists() and (root / "data").exists():
        task_dirs = [root]
    else:
        task_dirs = [
            p
            for p in root.iterdir()
            if p.is_dir() and (p / "meta").exists() and (p / "data").exists()
        ]

    if env_names:
        env_set = set(env_names)
        task_dirs = [p for p in task_dirs if p.name in env_set]

    return sorted(task_dirs, key=lambda p: p.name)


def _resolve_single_robocoin_task(dataset_path: str, env_names: Optional[List[str]]) -> Path:
    task_dirs = _discover_robocoin_task_dirs(Path(dataset_path).expanduser().resolve(), env_names)
    if not task_dirs:
        raise ValueError("No valid RoboCOIN task folder found.")
    if len(task_dirs) > 1:
        raise ValueError(
            "read/stats mode expects a single task folder. "
            "Pass a task path directly or specify one --env_name."
        )
    return task_dirs[0]


def inspect_robocoin_dataset_structure(
    dataset_path: str,
    env_names: Optional[List[str]] = None,
    max_tasks: int = 3,
    max_jsonl_lines: int = 2,
    max_chunks: int = 3,
    max_episode_files: int = 5,
    max_feature_keys: int = 30,
) -> Dict[str, Any]:
    root = Path(dataset_path).expanduser().resolve()
    task_dirs = _discover_robocoin_task_dirs(root, env_names)
    if not task_dirs:
        raise ValueError("No valid RoboCOIN task folders found.")

    report: Dict[str, Any] = {
        "dataset_path": str(root),
        "sample_limits": {
            "max_tasks": max_tasks,
            "max_jsonl_lines": max_jsonl_lines,
            "max_chunks": max_chunks,
            "max_episode_files_per_chunk": max_episode_files,
            "max_feature_keys": max_feature_keys,
        },
        "tasks": [],
    }

    for task_dir in task_dirs[: max(1, max_tasks)]:
        task_report: Dict[str, Any] = {
            "task_dir": str(task_dir),
            "task_name": task_dir.name,
        }

        meta_dir = task_dir / "meta"
        info_path = meta_dir / "info.json"
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as f:
                task_report["info_json"] = json.load(f)

        for jsonl_name in ("tasks.jsonl", "episodes.jsonl"):
            p = meta_dir / jsonl_name
            if p.exists():
                task_report[jsonl_name] = _inspect_jsonl(p, max_lines=max(1, max_jsonl_lines))

        data_dir = task_dir / "data"
        chunk_dirs = sorted([p for p in data_dir.glob("chunk-*") if p.is_dir()], key=lambda p: p.name)
        task_report["total_chunks"] = len(chunk_dirs)
        sampled_chunks = []
        for chunk_dir in chunk_dirs[: max(1, max_chunks)]:
            episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
            sampled_chunks.append(
                {
                    "chunk_dir": str(chunk_dir),
                    "episode_file_count": len(episode_files),
                    "sample_episode_files": [p.name for p in episode_files[: max(1, max_episode_files)]],
                }
            )
        task_report["sampled_chunks"] = sampled_chunks

        videos_dir = task_dir / "videos"
        if videos_dir.exists():
            video_files = sorted(videos_dir.glob("*.mp4"))
            task_report["video_file_count"] = len(video_files)
            task_report["sample_video_files"] = [p.name for p in video_files[: max(1, max_episode_files)]]

        if LeRobotDatasetMetadata is not None:
            try:
                meta = LeRobotDatasetMetadata(
                    task_dir.name,
                    str(task_dir),
                    CODEBASE_VERSION,
                    force_cache_sync=False,
                )
                feature_keys = sorted(list(meta.features.keys()))
                task_report["lerobot_metadata"] = {
                    "fps": float(meta.fps),
                    "total_episodes": int(meta.total_episodes),
                    "camera_keys": list(meta.camera_keys),
                    "feature_keys": feature_keys[: max(1, max_feature_keys)],
                    "feature_keys_truncated": len(feature_keys) > max(1, max_feature_keys),
                }
            except Exception as exc:
                task_report["lerobot_metadata_error"] = str(exc)
        else:
            task_report["lerobot_metadata_error"] = (
                "lerobot import failed; metadata summary unavailable"
            )

        report["tasks"].append(task_report)

    return report


class RoboCOINDatasetReader(Dataset):
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 50,
        fixed_action_dim: int = 32,
        pad_action_and_proprio: bool = True,
        use_proprio: bool = True,
        use_wrist_image: bool = True,
        use_num_images: Optional[int] = None,
        task_no_point: bool = True,
        state_key: Optional[str] = None,
        action_key: Optional[str] = None,
        prefer_eef_sim_pose: bool = False,
        delta_action: bool = False,
        num_episodes: Optional[object] = None,
        episode_index_start: Optional[int] = None,
        episode_index_end: Optional[int] = None,
        skip_images: bool = False,
        video_backend: str = "pyav",
        image_aug: bool = False,
    ):
        """
        Args:
            dataset_path: local path to RoboCOIN dataset root (LeRobot format).
            chunk_size: number of future actions to return.
            fixed_action_dim: pad action/proprio to this dim if enabled.
            pad_action_and_proprio: whether to pad action/proprio.
            use_proprio: whether to return proprio.
            use_wrist_image: whether to return images list (vs. image).
            use_num_images: optionally limit number of returned images.
            task_no_point: remove <point> tags from task text.
            state_key/action_key: override feature keys if needed.
            prefer_eef_sim_pose: prefer eef_sim_pose_* keys when present.
            delta_action: convert action to delta (action - state) except gripper.
            num_episodes: int/float/range string for random subset (see lerobot_datasets.py).
            episode_index_start/end: optional episode range filter.
            skip_images: if True, do not return images.
            video_backend: backend for LeRobotDataset video decoding.
            image_aug: enable simple image augmentations via lerobot ImageTransforms.
        """
        if LeRobotDataset is None:
            raise ImportError(
                "Failed to import lerobot. Install lerobot and retry."
            ) from _LEROBOT_IMPORT_ERROR

        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.fixed_action_dim = fixed_action_dim
        self.pad_action_and_proprio = pad_action_and_proprio
        self.use_proprio = use_proprio
        self.use_wrist_image = use_wrist_image
        self.use_num_images = use_num_images
        self.task_no_point = task_no_point
        self.delta_action = delta_action
        self.skip_images = skip_images

        dataset_meta = LeRobotDatasetMetadata(
            os.path.basename(dataset_path), dataset_path, CODEBASE_VERSION, force_cache_sync=False
        )
        self.camera_keys = list(dataset_meta.camera_keys)
        fps = dataset_meta.fps
        self.fps = fps

        features = dataset_meta.features
        # Resolve state/action keys
        if state_key is None:
            if prefer_eef_sim_pose and "eef_sim_pose_state" in features:
                state_key = "eef_sim_pose_state"
            elif "observation.state" in features:
                state_key = "observation.state"
            elif "state" in features:
                state_key = "state"
            elif "qpos" in features:
                state_key = "qpos"
            elif "eef_sim_pose_state" in features:
                state_key = "eef_sim_pose_state"
            else:
                raise ValueError(f"State key not found in dataset meta: {features}")
        if action_key is None:
            if prefer_eef_sim_pose and "eef_sim_pose_action" in features:
                action_key = "eef_sim_pose_action"
            elif "action" in features:
                action_key = "action"
            elif "actions" in features:
                action_key = "actions"
            elif "eef_sim_pose_action" in features:
                action_key = "eef_sim_pose_action"
            else:
                raise ValueError(f"Action key not found in dataset meta: {features}")

        self.state_key = state_key
        self.action_key = action_key

        # Build delta_timestamps
        delta_timestamps: Dict[str, List[float]] = {}
        if not self.skip_images:
            for key in self.camera_keys:
                delta_timestamps[key] = [0.0]
        delta_timestamps.update({
            self.state_key: [0.0],
            self.action_key: [t / fps for t in range(self.chunk_size)],
        })

        episodes = None
        if episode_index_start is not None or episode_index_end is not None:
            start = episode_index_start or 0
            end = episode_index_end if episode_index_end is not None else dataset_meta.total_episodes
            episodes = list(range(start, min(end, dataset_meta.total_episodes)))
        elif num_episodes is not None:
            # Reuse the same semantics as LeRobotDatasetWrapper
            np.random.seed(42)
            if isinstance(num_episodes, int):
                num_episodes = min(num_episodes, dataset_meta.total_episodes)
                episodes = sorted(
                    list(
                        np.random.choice(
                            dataset_meta.total_episodes, size=num_episodes, replace=False
                        )
                    )
                )
            elif isinstance(num_episodes, float):
                num_episodes = min(int(num_episodes * dataset_meta.total_episodes), dataset_meta.total_episodes)
                episodes = sorted(
                    list(
                        np.random.choice(
                            dataset_meta.total_episodes, size=num_episodes, replace=False
                        )
                    )
                )
            elif isinstance(num_episodes, str) and "(" in num_episodes and ")" in num_episodes:
                from_to = num_episodes.split("(")[1].split(")")[0].split(",")
                if len(from_to) != 2:
                    raise ValueError(f"Invalid num_episodes: {num_episodes}")
                from_episode = int(from_to[0])
                to_episode = min(int(from_to[1]), dataset_meta.total_episodes)
                episodes = list(range(from_episode, to_episode))
            else:
                raise ValueError(f"Invalid num_episodes: {num_episodes}")

        image_transforms = None
        if image_aug and ImageTransforms is not None:
            image_transforms_cfg = _build_image_transforms_cfg()
            if image_transforms_cfg is not None:
                image_transforms = ImageTransforms(image_transforms_cfg)
        if episodes is None:
            total = getattr(dataset_meta, "total_episodes", None)
            if total is None:
                # Fallback: count episode_*.parquet in default chunk dir
                data_dir = os.path.join(dataset_path, "data")
                chunk_dir = os.path.join(data_dir, "chunk-000")
                if os.path.isdir(chunk_dir):
                    total = len([f for f in os.listdir(chunk_dir) if f.startswith("episode_") and f.endswith(".parquet")])
            if total is None:
                raise ValueError("Unable to infer total episodes for RoboCOIN dataset.")
            episodes = list(range(int(total)))

        ds_kwargs = {
            "repo_id": os.path.basename(dataset_path),
            "root": dataset_path,
            "delta_timestamps": delta_timestamps,
            "video_backend": video_backend,
            "image_transforms": image_transforms,
            "check_timestamps": False,
            "episodes": episodes,
        }
        self.dataset = LeRobotDataset(**ds_kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.get(item, np.random)

    def get(self, item, rng=None):
        if isinstance(item, (int, np.integer)) and 0 <= int(item) < len(self.dataset):
            idx = int(item)
        else:
            raise ValueError(f"Invalid item: {item}")
        data_item = self.dataset[idx]

        images: List[np.ndarray] = []
        if not self.skip_images:
            for camera_key in self.camera_keys:
                if camera_key not in data_item:
                    continue
                cam = _to_numpy(data_item[camera_key])
                img = _format_image(cam)
                if img is not None:
                    images.append(img)
            if self.use_num_images is not None:
                images = images[: self.use_num_images]

        action = _to_numpy(data_item[self.action_key]).astype(np.float32)
        state = _to_numpy(data_item[self.state_key]).astype(np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if state.ndim == 1:
            state = state[None, :]

        if self.delta_action and action.ndim == 2 and state.ndim == 2:
            action[:, :-1] = action[:, :-1] - state[:, :-1]

        if action.ndim == 2 and action.shape[0] != self.chunk_size:
            if action.shape[0] > self.chunk_size:
                action = action[: self.chunk_size]
            elif action.shape[0] > 0:
                last = action[-1:]
                pad = np.repeat(last, self.chunk_size - action.shape[0], axis=0)
                action = np.concatenate([action, pad], axis=0)
            else:
                action = np.zeros((self.chunk_size, action.shape[-1]), dtype=action.dtype)

        if self.pad_action_and_proprio:
            if action.shape[-1] < self.fixed_action_dim:
                pad_len = self.fixed_action_dim - action.shape[-1]
                action = np.pad(action, ((0, 0), (0, pad_len)), mode="constant")
            if state.shape[-1] < self.fixed_action_dim:
                pad_len = self.fixed_action_dim - state.shape[-1]
                state = np.pad(state, ((0, 0), (0, pad_len)), mode="constant")

        instruction = data_item.get("task", "")
        if self.task_no_point and isinstance(instruction, str):
            instruction = re.sub(r"<point[^>]*>.*?</point>\\.", "", instruction)

        action_pad_mask = np.zeros_like(action, dtype=bool)

        ts = _to_scalar(data_item.get("timestamp", None))
        if ts is None:
            ts = _to_scalar(data_item.get("frame_index", None))

        metadata: Dict[str, Any] = {
            "timestamp": ts,
            "frame_index": _to_scalar(data_item.get("frame_index", None)),
            "episode_index": _to_scalar(data_item.get("episode_index", None)),
            "index": _to_scalar(data_item.get("index", None)),
            "task_index": _to_scalar(data_item.get("task_index", None)),
            "task": data_item.get("task", None),
            "file_path": self.dataset_path,
        }

        # Optional extra RoboCOIN metadata if present
        for key in (
            "subtask",
            "subtasks",
            "scene",
            "scene_desc",
            "scene_description",
            "motion",
            "motion_desc",
            "action_text",
            "instruction",
            "embodiment",
            "robot_id",
        ):
            if key in data_item and data_item[key] is not None:
                metadata[key] = data_item[key]

        return {
            "question": instruction,
            "timestep": ts,
            "answer": "Action",
            "style": "action",
            "action": action,
            "action_pad_mask": action_pad_mask,
            "proprio": state if self.use_proprio else None,
            "images": images if self.use_wrist_image else [images[0]] if images else [],
            "metadata": metadata,
        }

    def compute_stats(self) -> Dict[str, Any]:
        if hasattr(self.dataset, "meta") and getattr(self.dataset.meta, "stats", None) is not None:
            return self.dataset.meta.stats
        raise ValueError("Stats not available; compute stats with lerobot utilities.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboCOIN dataset reader.")
    parser.add_argument("--mode", choices=["read", "stats", "inspect"], required=True)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="RoboCOIN root path or a single task path")
    parser.add_argument("--env_name", type=str, nargs="+",
                        help="Task folder name(s). For read/stats, exactly one task must be selected.")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--fixed_action_dim", type=int, default=32)
    parser.add_argument("--prefer_eef_sim_pose", action="store_true")
    parser.add_argument("--delta_action", action="store_true")
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--video_backend", type=str, default="pyav")
    parser.add_argument("--max_items", type=int, default=3,
                        help="Number of samples to print in read mode")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional output JSON path for stats/inspect mode")
    parser.add_argument("--inspect_max_tasks", type=int, default=3,
                        help="Max tasks to inspect")
    parser.add_argument("--inspect_max_jsonl_lines", type=int, default=2,
                        help="Max lines sampled per jsonl in inspect mode")
    parser.add_argument("--inspect_max_chunks", type=int, default=3,
                        help="Max chunk folders to inspect")
    parser.add_argument("--inspect_max_episode_files", type=int, default=5,
                        help="Max episode parquet files to list per chunk")
    parser.add_argument("--inspect_max_feature_keys", type=int, default=30,
                        help="Max feature keys to list from lerobot metadata")
    args = parser.parse_args()

    if args.mode == "inspect":
        report = inspect_robocoin_dataset_structure(
            dataset_path=args.dataset_path,
            env_names=args.env_name,
            max_tasks=args.inspect_max_tasks,
            max_jsonl_lines=args.inspect_max_jsonl_lines,
            max_chunks=args.inspect_max_chunks,
            max_episode_files=args.inspect_max_episode_files,
            max_feature_keys=args.inspect_max_feature_keys,
        )
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        print(payload)
        if args.output_path:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(payload)
        return

    task_path = _resolve_single_robocoin_task(args.dataset_path, args.env_name)
    ds = RoboCOINDatasetReader(
        dataset_path=str(task_path),
        chunk_size=args.chunk_size,
        fixed_action_dim=args.fixed_action_dim,
        prefer_eef_sim_pose=args.prefer_eef_sim_pose,
        delta_action=args.delta_action,
        skip_images=args.skip_images,
        video_backend=args.video_backend,
    )

    if args.mode == "stats":
        stats = ds.compute_stats()
        payload = json.dumps(stats, ensure_ascii=False, indent=2)
        print(payload)
        if args.output_path:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(payload)
        return

    # read mode
    print(f"Dataset length: {len(ds)}")
    for i in range(min(args.max_items, len(ds))):
        item = ds[i]
        print(f"[{i}] question={item['question']}")
        print(f"  action: {item['action'].shape}")
        print(f"  proprio: {item['proprio'].shape if item['proprio'] is not None else None}")
        print(f"  images: {len(item['images'])}")
        print(f"  metadata: {item['metadata']}")


if __name__ == "__main__":
    main()
