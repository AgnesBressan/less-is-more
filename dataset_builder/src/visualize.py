"""Visualize Grand Tour camera images and elevation maps.

Pulls the required topics from HuggingFace (if not already local), then
iterates through frames showing a matplotlib figure per frame:

  Without --with-side-cams:
      [front camera]  |  [elevation map]

  With --with-side-cams:
      [left cam]  [front cam]  [right cam]  |  [elevation map]

Usage
-----
  uv run dataset_builder/src/visualize.py \\
      --mission-timestamp 2024-10-01-11-29-55 \\
      --dataset-dir data/dataset \\
      [--with-side-cams] \\
      [--every 50] \\
      [--max-frames 200] \\
      [--save-dir /tmp/vis]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.grandtour_hub import HF_REVISION_MAIN, pull_mission_topics
from dataset_builder.src.mission_data_source import GrandTourZarrSource

BUILDER_TOPICS = ["hdr_front", "dlio_map_odometry", "elevation_map"]
SIDE_CAM_TOPICS = ["hdr_left", "hdr_right"]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_elevation(ax, elev: np.ndarray, title: str = "Elevation map") -> None:
    """Render a (G, G) float32 elevation array (NaN = unknown) on ax."""
    # Flip so robot-forward points upwards in the plot
    display = np.flipud(elev)
    vmin = np.nanpercentile(display, 5)
    vmax = np.nanpercentile(display, 95)

    cmap = plt.cm.terrain.copy()
    cmap.set_bad(color="lightgrey")

    im = ax.imshow(
        display,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="z [m]")

    cy, cx = display.shape[0] // 2, display.shape[1] // 2
    ax.plot(cx, cy, "r^", markersize=8, label="robot")


def render_frame(
    source: GrandTourZarrSource,
    frame_idx: int,
    with_side_cams: bool,
    save_path: Path | None = None,
) -> None:
    img_front = source.get_image(frame_idx)
    elev = source.get_elevation(frame_idx)
    ts = source.get_timestamp(frame_idx)

    has_side = with_side_cams

    if has_side:
        mission_dir = source.mission_dir
        img_l = np.array(Image.open(mission_dir / "images" / "hdr_left" / f"{frame_idx:06d}.jpeg").convert("RGB"))
        img_r = np.array(Image.open(mission_dir / "images" / "hdr_right" / f"{frame_idx:06d}.jpeg").convert("RGB"))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_l)
        axes[0].set_title("hdr_left", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(img_front)
        axes[1].set_title("hdr_front", fontsize=9)
        axes[1].axis("off")
        axes[2].imshow(img_r)
        axes[2].set_title("hdr_right", fontsize=9)
        axes[2].axis("off")
        elev_ax = axes[3]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_front)
        axes[0].set_title("hdr_front", fontsize=9)
        axes[0].axis("off")
        elev_ax = axes[1]

    _plot_elevation(elev_ax, elev)

    fig.suptitle(f"frame {frame_idx}  |  t={ts:.3f}", fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Grand Tour camera + elevation maps.")
    p.add_argument("--mission-timestamp", required=True,
                   help="Mission folder name, e.g. 2024-10-01-11-29-55")
    p.add_argument("--dataset-dir", default="data/dataset",
                   help="Root dataset directory (grandtour/ subfolder will be used).")
    p.add_argument("--with-side-cams", action="store_true",
                   help="Also pull and display hdr_left and hdr_right.")
    p.add_argument("--every", type=int, default=50,
                   help="Visualize every N-th frame (default: 50).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many camera frames (before the --every filter).")
    p.add_argument("--revision", default=HF_REVISION_MAIN,
                   help="HuggingFace repo revision to pull from.")
    p.add_argument("--save-dir", default=None,
                   help="If set, save PNG files here instead of showing interactively.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    grandtour_dir = dataset_dir / "grandtour"
    grandtour_dir.mkdir(parents=True, exist_ok=True)

    topics = BUILDER_TOPICS + (SIDE_CAM_TOPICS if args.with_side_cams else [])
    print(f"Pulling topics {topics} for mission {args.mission_timestamp} …")
    pull_mission_topics(
        missions=[args.mission_timestamp],
        topics=topics,
        dataset_folder=grandtour_dir,
        revision=args.revision,
        skip_existing=True,
    )

    mission_dir = grandtour_dir / args.mission_timestamp
    source = GrandTourZarrSource(mission_dir)
    print(f"Mission has {len(source)} camera frames.")

    save_dir = Path(args.save_dir) if args.save_dir else None
    n_total = min(len(source), args.max_frames) if args.max_frames else len(source)
    rendered = 0

    for frame_idx in range(n_total):
        if frame_idx % args.every != 0:
            continue

        save_path = save_dir / f"frame_{frame_idx:06d}.png" if save_dir else None
        render_frame(source, frame_idx, args.with_side_cams, save_path)
        rendered += 1
        print(f"  rendered frame {frame_idx}/{n_total - 1}")

    print(f"Done — rendered {rendered} frames.")


if __name__ == "__main__":
    main()
