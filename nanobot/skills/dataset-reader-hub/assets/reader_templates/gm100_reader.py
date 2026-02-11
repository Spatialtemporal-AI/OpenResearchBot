"""
GM100 dataset reader (LeRobot).

Returns the same dict structure as rc_reader.py and robocoin_reader.py:
  - question, timestep, answer, style, action, action_pad_mask, proprio, images, metadata
"""

from __future__ import annotations

import argparse
import json
import os
import re
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import inspect

import numpy as np

from olmo.data.dataset import Dataset

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


PREFERRED_VIDEO_KEYS = [
    "observation.images.camera_top",
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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


def _order_camera_keys(camera_keys: List[str]) -> List[str]:
    ordered: List[str] = []
    for k in PREFERRED_VIDEO_KEYS:
        if k in camera_keys:
            ordered.append(k)
    remaining = [k for k in camera_keys if k not in ordered]
    ordered.extend(sorted(remaining))
    return ordered


def _to_2d(arr: Any) -> np.ndarray:
    arr = _to_numpy(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


class GM100DatasetReader(Dataset):
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
        state_keys: Optional[List[str]] = None,
        action_keys: Optional[List[str]] = None,
        include_velocity: bool = False,
        include_effort: bool = False,
        num_episodes: Optional[object] = None,
        episode_index_start: Optional[int] = None,
        episode_index_end: Optional[int] = None,
        env_names: Optional[List[str]] = None,
        skip_images: bool = False,
        video_backend: str = "pyav",
        image_aug: bool = False,
    ):
        """
        Args:
            dataset_path: path to GM100 root or a single task folder.
            chunk_size: number of future actions to return.
            fixed_action_dim: pad action/proprio to this dimension.
            pad_action_and_proprio: whether to pad action/proprio.
            use_proprio: whether to return proprio.
            use_wrist_image: whether to return images list (vs. image).
            use_num_images: optionally limit number of returned images.
            task_no_point: remove <point> tags from task text.
            state_keys/action_keys: override default GM100 keys (lists).
            include_velocity: include arm velocity in state.
            include_effort: include arm/effector effort in state.
            num_episodes: int/float/range string for random subset (see lerobot_datasets.py).
            episode_index_start/end: optional episode range filter.
            env_names: list of task folder names to include (if dataset_path is a root).
            skip_images: if True, do not return images.
            video_backend: backend for LeRobotDataset video decoding.
            image_aug: enable simple image augmentations via lerobot ImageTransforms.
        """
        if LeRobotDataset is None:
            raise ImportError(
                "Failed to import lerobot. Install lerobot and retry."
            ) from _LEROBOT_IMPORT_ERROR

        self.dataset_path = Path(dataset_path)
        self.chunk_size = chunk_size
        self.fixed_action_dim = fixed_action_dim
        self.pad_action_and_proprio = pad_action_and_proprio
        self.use_proprio = use_proprio
        self.use_wrist_image = use_wrist_image
        self.use_num_images = use_num_images
        self.task_no_point = task_no_point
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.include_velocity = include_velocity
        self.include_effort = include_effort
        self.skip_images = skip_images
        self.video_backend = video_backend

        image_transforms = None
        if image_aug and ImageTransforms is not None:
            cfg = _build_image_transforms_cfg()
            if cfg is not None:
                image_transforms = ImageTransforms(cfg)

        self._datasets: List[Dict[str, Any]] = []
        self._cum_lengths: List[int] = []

        task_dirs = self._discover_task_dirs(env_names)
        if not task_dirs:
            raise ValueError("No valid GM100 tasks found under dataset_path.")

        for task_dir in task_dirs:
            meta = self._load_task_meta(task_dir)
            dataset_meta = LeRobotDatasetMetadata(
                os.path.basename(str(task_dir)), str(task_dir), CODEBASE_VERSION, force_cache_sync=False
            )
            camera_keys = _order_camera_keys(list(dataset_meta.camera_keys))
            fps = float(dataset_meta.fps)

            features = dataset_meta.features
            state_keys, action_keys = self._resolve_state_action_keys(features)

            delta_timestamps: Dict[str, List[float]] = {}
            if not self.skip_images:
                for key in camera_keys:
                    delta_timestamps[key] = [0.0]
            for key in state_keys:
                delta_timestamps[key] = [0.0]
            for key in action_keys:
                delta_timestamps[key] = [t / fps for t in range(self.chunk_size)]

            episodes = None
            if episode_index_start is not None or episode_index_end is not None:
                start = episode_index_start or 0
                end = episode_index_end if episode_index_end is not None else dataset_meta.total_episodes
                episodes = list(range(start, min(end, dataset_meta.total_episodes)))
            elif num_episodes is not None:
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
                    num_episodes = int(dataset_meta.total_episodes * num_episodes)
                    episodes = sorted(
                        list(
                            np.random.choice(
                                dataset_meta.total_episodes, size=num_episodes, replace=False
                            )
                        )
                    )
                elif isinstance(num_episodes, str):
                    if ":" in num_episodes:
                        parts = num_episodes.split(":")
                        start = int(parts[0]) if parts[0] else 0
                        end = int(parts[1]) if len(parts) > 1 and parts[1] else dataset_meta.total_episodes
                        episodes = list(range(start, min(end, dataset_meta.total_episodes)))
                    elif "-" in num_episodes:
                        start_s, end_s = num_episodes.split("-")
                        start = int(start_s)
                        end = int(end_s)
                        episodes = list(range(start, min(end, dataset_meta.total_episodes)))
            if episodes is None:
                episodes = list(range(int(dataset_meta.total_episodes)))

            ds_kwargs = {
                "repo_id": os.path.basename(str(task_dir)),
                "root": str(task_dir),
                "delta_timestamps": delta_timestamps,
                "video_backend": video_backend,
                "image_transforms": image_transforms,
                "check_timestamps": False,
                "episodes": episodes,
            }
            dataset = LeRobotDataset(**ds_kwargs)

            entry = {
                "dataset": dataset,
                "camera_keys": camera_keys,
                "state_keys": state_keys,
                "action_keys": action_keys,
                "prompt": meta["prompt"],
                "robot_type": meta["robot_type"],
                "task_dir": str(task_dir),
            }
            self._datasets.append(entry)

        total = 0
        for entry in self._datasets:
            total += len(entry["dataset"])
            self._cum_lengths.append(total)

    def _discover_task_dirs(self, env_names: Optional[List[str]]) -> List[Path]:
        if (self.dataset_path / "meta" / "info.json").exists():
            task_dirs = [self.dataset_path]
        else:
            task_dirs = [
                p for p in self.dataset_path.iterdir() if p.is_dir() and (p / "meta" / "info.json").exists()
            ]
        if env_names:
            env_set = set(env_names)
            task_dirs = [p for p in task_dirs if p.name in env_set]
        return sorted(task_dirs, key=lambda p: p.name)

    def _load_task_meta(self, task_dir: Path) -> Dict[str, Any]:
        info = load_json(task_dir / "meta" / "info.json")
        tasks_meta = load_jsonl(task_dir / "meta" / "tasks.jsonl")
        episodes_meta = load_jsonl(task_dir / "meta" / "episodes.jsonl")
        task_prompt = None
        if tasks_meta:
            task_prompt = tasks_meta[0].get("task")
        if not task_prompt and episodes_meta:
            tlist = episodes_meta[0].get("tasks", [])
            if tlist:
                task_prompt = tlist[0]
        return {
            "prompt": task_prompt or task_dir.name,
            "robot_type": info.get("robot_type", "unknown"),
        }

    def _resolve_state_action_keys(self, features: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        if self.state_keys:
            state_keys = list(self.state_keys)
        else:
            state_keys = []
            if "observation.state.arm.position" in features:
                state_keys.append("observation.state.arm.position")
            if "observation.state.effector.position" in features:
                state_keys.append("observation.state.effector.position")
            if self.include_velocity and "observation.state.arm.velocity" in features:
                state_keys.append("observation.state.arm.velocity")
            if self.include_effort:
                if "observation.state.arm.effort" in features:
                    state_keys.append("observation.state.arm.effort")
                if "observation.state.effector.effort" in features:
                    state_keys.append("observation.state.effector.effort")
            if not state_keys:
                state_keys = sorted([k for k in features.keys() if k.startswith("observation.state")])

        if self.action_keys:
            action_keys = list(self.action_keys)
        else:
            action_keys = []
            if "action.arm.position" in features:
                action_keys.append("action.arm.position")
            if "action.effector.position" in features:
                action_keys.append("action.effector.position")
            if not action_keys:
                action_keys = sorted([k for k in features.keys() if k.startswith("action")])

        if not state_keys:
            raise ValueError("No valid state keys found for GM100 dataset.")
        if not action_keys:
            raise ValueError("No valid action keys found for GM100 dataset.")
        return state_keys, action_keys

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.get(item, np.random)

    def get(self, item, rng=None):
        if not isinstance(item, (int, np.integer)):
            raise ValueError(f"Invalid item: {item}")
        idx = int(item)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        ds_idx = bisect_left(self._cum_lengths, idx + 1)
        prev = 0 if ds_idx == 0 else self._cum_lengths[ds_idx - 1]
        local_idx = idx - prev
        entry = self._datasets[ds_idx]
        data_item = entry["dataset"][local_idx]

        images: List[np.ndarray] = []
        if not self.skip_images:
            for camera_key in entry["camera_keys"]:
                if camera_key not in data_item:
                    continue
                cam = _to_numpy(data_item[camera_key])
                img = _format_image(cam)
                if img is not None:
                    images.append(img)
            if self.use_num_images is not None:
                images = images[: self.use_num_images]

        action = self._collect_action(data_item, entry["action_keys"])
        state = self._collect_state(data_item, entry["state_keys"]) if self.use_proprio else None

        if action.ndim == 1:
            action = action[None, :]
        if state is not None and state.ndim == 1:
            state = state[None, :]

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
            if state is not None and state.shape[-1] < self.fixed_action_dim:
                pad_len = self.fixed_action_dim - state.shape[-1]
                state = np.pad(state, ((0, 0), (0, pad_len)), mode="constant")

        instruction = data_item.get("task", "")
        if not instruction:
            instruction = entry["prompt"]
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
            "task": instruction,
            "file_path": entry["task_dir"],
            "embodiment": entry.get("robot_type", "unknown"),
        }

        return {
            "question": instruction,
            "timestep": ts,
            "answer": "Action",
            "style": "action",
            "action": action,
            "action_pad_mask": action_pad_mask,
            "proprio": state,
            "images": images if self.use_wrist_image else [images[0]] if images else [],
            "metadata": metadata,
        }

    def _collect_state(self, data_item: Dict[str, Any], keys: List[str]) -> np.ndarray:
        arrays: List[np.ndarray] = []
        for key in keys:
            if key not in data_item:
                raise KeyError(f"Missing state key in sample: {key}")
            arr = _to_2d(data_item[key]).astype(np.float32)
            arrays.append(arr[:1])
        if not arrays:
            raise ValueError("No state arrays found for GM100 sample.")
        return np.concatenate(arrays, axis=-1)

    def _collect_action(self, data_item: Dict[str, Any], keys: List[str]) -> np.ndarray:
        arrays: List[np.ndarray] = []
        for key in keys:
            if key not in data_item:
                raise KeyError(f"Missing action key in sample: {key}")
            arr = _to_2d(data_item[key]).astype(np.float32)
            arrays.append(arr)
        if not arrays:
            raise ValueError("No action arrays found for GM100 sample.")
        min_len = min(a.shape[0] for a in arrays)
        arrays = [a[:min_len] for a in arrays]
        return np.concatenate(arrays, axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GM100 dataset reader (LeRobot).")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--fixed_action_dim", type=int, default=32)
    parser.add_argument("--include_velocity", action="store_true")
    parser.add_argument("--include_effort", action="store_true")
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--max_items", type=int, default=3)
    args = parser.parse_args()

    ds = GM100DatasetReader(
        dataset_path=args.dataset_path,
        chunk_size=args.chunk_size,
        fixed_action_dim=args.fixed_action_dim,
        include_velocity=args.include_velocity,
        include_effort=args.include_effort,
        skip_images=args.skip_images,
    )

    print(f"Dataset length: {len(ds)}")
    for i in range(min(args.max_items, len(ds))):
        item = ds[i]
        print(f"[{i}] question={item['question']}")
        print(f"  action: {item['action'].shape}  proprio: {item['proprio'].shape if item['proprio'] is not None else None}  images: {len(item['images'])}")
