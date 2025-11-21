import argparse
from pathlib import Path
import os

import numpy as np

# Adjust this import path if your project layout differs
from config_files import config_copy
from config_files import map_loader


def main():
    parser = argparse.ArgumentParser(
        description="Compute a reference centerline .npy from a ghost / replay .gbx file."
    )
    parser.add_argument(
        "--replay_path",
        type=str,
        required=True,
        help=(
            "Path to the replay / ghost .gbx file.\n"
            "Example: 'C:/.../TrackMania Nations Forever/Tracks/Replays/Autosaves/MyReplay.Replay.Gbx'"
        ),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help=(
            "Name of the output .npy file to create inside the 'maps' folder.\n"
            "Example: 'level1_0.5m_cl.npy'"
        ),
    )
    parser.add_argument(
        "--densify_factor",
        type=int,
        default=1,
        help=(
            "How many times to densify the raw ghost path.\n"
            "1 = no densification (use raw records).\n"
            "Higher values create more points between original samples."
        ),
    )

    args = parser.parse_args()

    maps_dir = Path(config_copy.windows_project_path / "Maps")

    replay_path = Path(args.replay_path)

    print(f"Loading ghost positions from: {replay_path}")
    raw_positions_list = map_loader.gbx_to_raw_pos_list(replay_path)

    print(f"Got {len(raw_positions_list)} raw positions from ghost.")

    if args.densify_factor > 1:
        print(f"Densifying path with factor {args.densify_factor} ...")
        raw_positions_list = map_loader.densify_raw_pos_list_n_times(
            raw_positions_list, args.densify_factor
        )
        print(f"After densification: {len(raw_positions_list)} positions.")

    zone_centers = np.array(raw_positions_list, dtype=np.float32)

    output_path = maps_dir / args.output_name
    np.save(str(output_path), zone_centers)

    print(f"Saved reference line to: {output_path}")


if __name__ == "__main__":
    main()
