"""Generate camera parameters for a multi-view orbit. Tier 1.5 item 5.

Produces N camera (yaw, pitch, distance, fov) tuples sampled around the scene.
The orbit is two parameters x_yaw_grid : N_az points around the y-axis, plus
N_el elevation steps (e.g. -10, 5, 20 degrees).

Output: a JSON file the run-all driver can iterate through.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple


def make_orbit_cameras(
    n_azimuth: int = 16,
    elevations: Tuple[float, ...] = (-2.0, 18.0, 35.0),
    distance: float = 1.4,
    fov: float = 42.0,
    azimuth_offset: float = 0.0,
) -> List[dict]:
    """Return list of camera-param dicts."""
    cams = []
    for el in elevations:
        for i in range(n_azimuth):
            az = azimuth_offset + (360.0 * i / n_azimuth)
            cams.append(dict(
                yaw_deg=float(az),
                pitch_deg=float(el),
                distance=float(distance),
                fov_deg=float(fov),
                cam_id=f"az{i:02d}_el{int(el):+d}",
            ))
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-azimuth", type=int, default=16)
    ap.add_argument("--elevations", type=float, nargs="+", default=[-2.0, 18.0, 35.0])
    ap.add_argument("--distance", type=float, default=1.4)
    ap.add_argument("--fov", type=float, default=42.0)
    ap.add_argument("--out", type=Path, default=Path("multiview_cameras.json"))
    args = ap.parse_args()

    cams = make_orbit_cameras(args.n_azimuth, tuple(args.elevations), args.distance, args.fov)
    args.out.write_text(json.dumps(cams, indent=2))
    print(f"wrote {len(cams)} cameras to {args.out}")


if __name__ == "__main__":
    main()
