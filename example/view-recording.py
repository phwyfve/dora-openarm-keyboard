"""Export an episode to Rerun without resampling its recorded streams."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb


def main() -> None:
    """Write a recording that can be opened with the Rerun viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", default="0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episode = args.dataset / "episodes" / args.episode
    if not episode.is_dir():
        parser.error(f"Episode directory does not exist: {episode}")
    output = args.output or args.dataset / f"episode_{args.episode}.rrd"
    if output.exists():
        parser.error(f"Output already exists: {output}")
    paths = sorted(episode.glob("*/arms/*/state.parquet"))
    cameras = sorted((episode / "cameras").glob("*"))
    cameras = [camera for camera in cameras if camera.is_dir()]
    if not paths or not cameras:
        parser.error("Expected arm state Parquet files and camera directories")

    rr.init(f"OpenArm - episode {args.episode}")
    rr.save(output)
    camera_views = [
        rrb.Spatial2DView(origin=f"cameras/{camera.name}", name=camera.name)
        for camera in cameras
    ]
    curve_views = [
        rrb.TimeSeriesView(origin=f"{kind}/{side}/{field}", name=f"{kind} {side} {field}")
        for side in ("left", "right")
        for kind, field in (
            ("action", "qpos"),
            ("obs", "qpos"),
            ("obs", "qvel"),
            ("obs", "qtorque"),
        )
    ]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Grid(*camera_views, grid_columns=2),
                rrb.Tabs(*curve_views),
                column_shares=[0.65, 0.35],
            ),
            rrb.TimePanel(state="expanded"),
        )
    )
    try:
        for path in paths:
            kind, _, side, _ = path.relative_to(episode).parts
            table = pq.read_table(path)
            times = table["timestamp"].cast("int64").to_numpy().astype("datetime64[ns]")
            for field in ("qpos", "qvel", "qtorque"):
                if field not in table.column_names:
                    continue
                values = np.asarray(table[field].to_pylist())
                if values.shape != (len(times), 8) or not np.isfinite(values).all():
                    raise ValueError(f"Expected finite shape-({len(times)}, 8) {path}:{field}")
                for joint in range(8):
                    name = f"joint{joint + 1}" if joint < 7 else "gripper"
                    rr.send_columns(
                        f"{kind}/{side}/{field}/{name}",
                        indexes=[rr.TimeColumn("timestamp", timestamp=times)],
                        columns=rr.Scalars.columns(scalars=values[:, joint]),
                    )
            print(f"{path.relative_to(episode)}: {len(times)} samples", flush=True)

        for camera in cameras:
            images = sorted(camera.glob("*.jpeg"))
            for image in images:
                rr.set_time("timestamp", timestamp=np.datetime64(int(image.stem), "ns"))
                rr.log(f"cameras/{camera.name}", rr.EncodedImage(path=image))
            print(f"{camera.name}: {len(images)} original images", flush=True)
    finally:
        rr.disconnect()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
