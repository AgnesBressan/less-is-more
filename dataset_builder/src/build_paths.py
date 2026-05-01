"""Build path zarr groups from Grand Tour missions.

Supports two dataset types:
  geo — MPPI-planned geometric paths (D_geo)
  tel — Imitation paths from the robot's recorded trajectory (D_tel)

Both types write the same zarr schema under data/{geometric,teleop}_paths/.

Usage
-----
  uv run dataset_builder/src/build_paths.py dataset_type=geo
  uv run dataset_builder/src/build_paths.py dataset_type=tel
  uv run dataset_builder/src/build_paths.py --config-name build_debug dataset_type=geo
"""

import csv
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

from dataset_builder.helpers.transform_helpers import transform_se2_odom_to_base, convert_se2_to_transform
from dataset_builder.mppi_planner.mppi_planner import GridMap2D, MPPIPlanner
from dataset_builder.src.mission_data_source import GrandTourZarrSource
from utils.grandtour_hub import HF_REVISION_MAIN, HF_REVISION_LIMO, pull_mission_topics

log = logging.getLogger(__name__)

# Topics required per dataset_type
_TOPICS_GEO = ["hdr_front", "dlio_map_odometry", "elevation_map"]
_TOPICS_TEL = ["hdr_front", "dlio_map_odometry"]

# D_geo goal distribution: forward-biased Gaussian in base frame
_GOAL_MEAN = np.array([5.0, 0.0, 0.0])
_GOAL_COV = np.diag([2.5**2, 2.0**2, (np.pi / 4) ** 2])


def _resolve_paths(cfg: DictConfig) -> None:
    root = Path(__file__).resolve().parents[2]
    for key in ("dataset_folder", "missions_csv"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(root / p)


def _parse_missions_csv(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["Timestamp"].strip() for row in csv.DictReader(f) if row.get("Timestamp", "").strip()]


def _write_zarr(zarr_dir: Path, paths: list, goals: list, image_ids: list, goal_times: list) -> None:
    zarr_dir.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(zarr_dir), mode="w")
    g.create_dataset("path",      data=np.array(paths,      dtype=np.float32), chunks=(1000, 50, 3))
    g.create_dataset("goal",      data=np.array(goals,      dtype=np.float32), chunks=(1000, 3))
    g.create_dataset("image_id",  data=np.array(image_ids,  dtype=np.int64),   chunks=(1000,))
    g.create_dataset("goal_time", data=np.array(goal_times, dtype=np.float32), chunks=(1000,))


# ── D_geo helpers ─────────────────────────────────────────────────────────────

def _sample_geo_goal(rng: np.random.Generator) -> np.ndarray:
    pose = rng.multivariate_normal(_GOAL_MEAN, _GOAL_COV).astype(np.float32)
    pose[0] = abs(pose[0])
    return pose


def _check_min_nan_dist(elev: np.ndarray, min_cells: int) -> bool:
    t = torch.from_numpy(elev)
    nan_mask = torch.isnan(t)
    if not nan_mask.any():
        return True
    H, W = t.shape
    rows, cols = torch.where(nan_mask)
    dist = torch.sqrt(((rows - H // 2).float() ** 2 + (cols - W // 2).float() ** 2))
    return dist.min().item() >= min_cells


def _build_geo_mission(
    source: GrandTourZarrSource,
    mission_dir: Path,
    planner: MPPIPlanner,
    cfg: DictConfig,
    rng: np.random.Generator,
    device: str,
) -> int:
    origin = torch.tensor([-cfg.map_size, -cfg.map_size], dtype=torch.float32, device=device)
    start = torch.zeros(3, dtype=torch.float32, device=device)
    paths_list, goals_list, image_ids_list, goal_times_list = [], [], [], []

    for i in tqdm(range(len(source)), desc=mission_dir.name, leave=False):
        elev_np = source.get_elevation(i)

        if np.isnan(elev_np).mean() > cfg.max_nan_frac:
            continue
        if not _check_min_nan_dist(elev_np, cfg.min_nan_dist_cells):
            continue

        elev = torch.from_numpy(elev_np).to(device)
        # GridMap2D indexed as [x_idx, y_idx] so we transpose the [y, x] arrays
        # Traversability is computed internally by MPPIObjective via the filter
        gm = GridMap2D(
            elevation=elev.T,
            resolution=cfg.map_resolution,
            origin_xy=origin,
        )

        for _ in range(cfg.paths_per_image):
            goal = _sample_geo_goal(rng)
            states = planner.plan(gm, start, torch.from_numpy(goal).to(device))
            paths_list.append(states.cpu().numpy().astype(np.float32))
            goals_list.append(goal)
            image_ids_list.append(i)
            goal_times_list.append(float(cfg.goal_time))

    n = len(paths_list)
    if n > 0:
        _write_zarr(mission_dir / "data" / "geometric_paths",
                    paths_list, goals_list, image_ids_list, goal_times_list)
    return n


# ── D_tel helpers ─────────────────────────────────────────────────────────────

def _build_tel_mission(
    source: GrandTourZarrSource,
    mission_dir: Path,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> int:
    paths_list, goals_list, image_ids_list, goal_times_list = [], [], [], []

    for i in tqdm(range(len(source)), desc=mission_dir.name, leave=False):
        goal_time = abs(float(rng.normal(cfg.goal_time_mean, cfg.goal_time_std)))
        goal_time = max(goal_time, 0.5)  # at least 0.5 s into the future

        traj_world = source.get_trajectory_world(i, duration=goal_time, n=50)
        pose_world = source.get_pose_se2_world(i)
        T_world_base = convert_se2_to_transform(pose_world)

        path_base = transform_se2_odom_to_base(traj_world, T_world_base)
        goal_base = path_base[-1]

        # Skip stationary segments
        if np.linalg.norm(path_base[0, :2] - path_base[-1, :2]) < 0.1:
            continue

        paths_list.append(path_base.astype(np.float32))
        goals_list.append(goal_base.astype(np.float32))
        image_ids_list.append(i)
        goal_times_list.append(float(goal_time))

    n = len(paths_list)
    if n > 0:
        _write_zarr(mission_dir / "data" / "teleop_paths",
                    paths_list, goals_list, image_ids_list, goal_times_list)
    return n


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../configs", config_name="build")
def main(cfg: DictConfig) -> None:
    _resolve_paths(cfg)

    dataset_type = cfg.get("dataset_type", "geo")
    assert dataset_type in ("geo", "tel"), f"dataset_type must be 'geo' or 'tel', got {dataset_type!r}"

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device

    missions = _parse_missions_csv(cfg.missions_csv)
    grandtour_dir = Path(cfg.dataset_folder) / "grandtour"
    grandtour_dir.mkdir(parents=True, exist_ok=True)

    if dataset_type == "geo":
        topics = _TOPICS_GEO
        revision = HF_REVISION_MAIN
        planner = MPPIPlanner(cfg.mppi, device)
    else:
        topics = _TOPICS_TEL
        revision = HF_REVISION_LIMO
        planner = None

    log.info(f"Building D_{dataset_type} paths for {len(missions)} mission(s)")
    total = 0
    for mission_ts in missions:
        mission_dir = grandtour_dir / mission_ts
        log.info(f"[{mission_ts}] pulling topics …")
        pull_mission_topics(
            missions=[mission_ts],
            topics=topics,
            dataset_folder=grandtour_dir,
            revision=revision,
            skip_existing=True,
        )

        source = GrandTourZarrSource(mission_dir, cfg.map_size, cfg.map_resolution)
        log.info(f"[{mission_ts}] {len(source)} frames")

        if dataset_type == "geo":
            n = _build_geo_mission(source, mission_dir, planner, cfg, rng, device)
        else:
            n = _build_tel_mission(source, mission_dir, cfg, rng)

        log.info(f"[{mission_ts}] wrote {n} samples")
        total += n

    log.info(f"Done — {total} total samples across {len(missions)} missions")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
