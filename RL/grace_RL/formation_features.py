"""
formation_features.py
---------------------
Reads tournament high-level CSV logs and produces a numpy feature matrix for
formation clustering.

Feature vector (28 dims per datapoint):
  Ball (6):  px, py, pz, vx, vy, vz
  Car 0 (relative to own team, closest to own goal) (5):  px, py, pz, vx, vy
  Car 1 (5):  px, py, pz, vx, vy
  Car 2 (5):  px, py, pz, vx, vy
  Opponent summary (7):
    centroid px, py  (2)
    centroid vx, vy  (2)
    closest opponent dist to their goal  (1)
    median opponent dist to ball  (2: dx, dy)

All positions/velocities are expressed from the perspective of the "own" team
so that blue and orange are treated symmetrically:
  - Orange perspective: y-axis flipped (field is symmetric along y), x unchanged.

Subsampling: every SUBSAMPLE_EVERY rows (~0.25 s at 8-tick-sampled logs).

Outputs (saved to OUTFILE):
  features.npy  — float32 array of shape (N, 28)
  metadata.npy  — structured array with fields: match_id(i4), tick(i4), perspective(U6)
"""

import os
import glob
import numpy as np
import csv

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR       = os.path.join(os.path.dirname(__file__), "run0", "out", "tournament_high_logs")
OUT_DIR       = os.path.join(os.path.dirname(__file__), "run0", "out", "formation_clusters")
SUBSAMPLE_EVERY = 5       # read every Nth row after initial sort (≈0.25 s coverage)
BLUE_GOAL_Y   = -5120.0  # Rocket League standard
ORANGE_GOAL_Y =  5120.0

os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_DIM = 28

CAR_SLOTS_BLUE   = [f"blue-{i}"   for i in range(3)]
CAR_SLOTS_ORANGE = [f"orange-{i}" for i in range(3)]


def _dist_to_goal(px: float, py: float, goal_y: float) -> float:
    return float(np.sqrt(px**2 + (py - goal_y)**2))


def _car_xyz(row: dict, slot: str):
    """Return (px, py, pz, vx, vy) or None if slot is empty."""
    px = row.get(f"{slot}_px", "")
    if px == "" or px is None:
        return None
    return (
        float(px),
        float(row[f"{slot}_py"]),
        float(row[f"{slot}_pz"]),
        float(row[f"{slot}_vx"]),
        float(row[f"{slot}_vy"]),
    )


def _flip_y(vec):
    """Flip y-component for orange→blue perspective transform."""
    return (-vec[0], -vec[1]) if len(vec) == 2 else (vec[0], -vec[1], vec[2], vec[3], -vec[4])


def build_feature(row: dict, perspective: str):
    """
    Build a 28-dim feature vector for a given row and perspective ('blue'|'orange').
    Returns None if data is insufficient (missing cars).
    """
    # Ball
    ball_pos = np.array([float(row["ball_px"]), float(row["ball_py"]), float(row["ball_pz"])], dtype=np.float32)
    ball_vel = np.array([float(row["ball_vx"]), float(row["ball_vy"]), float(row["ball_vz"])], dtype=np.float32)

    if perspective == "blue":
        own_slots = CAR_SLOTS_BLUE
        opp_slots = CAR_SLOTS_ORANGE
        own_goal_y = BLUE_GOAL_Y
        opp_goal_y = ORANGE_GOAL_Y
        sign = 1.0
    else:
        own_slots = CAR_SLOTS_ORANGE
        opp_slots = CAR_SLOTS_BLUE
        own_goal_y = ORANGE_GOAL_Y
        opp_goal_y = BLUE_GOAL_Y
        sign = -1.0  # flip y so orange perspective mirrors blue

    # Own cars: sort by distance to own goal (nearest first)
    own_cars = []
    for slot in own_slots:
        c = _car_xyz(row, slot)
        if c is not None:
            d = _dist_to_goal(c[0], c[1], own_goal_y)
            own_cars.append((d, c))
    own_cars.sort(key=lambda x: x[0])

    if len(own_cars) < 3:
        return None  # skip incomplete states

    # Opponent cars: sort by distance to opponent goal
    opp_cars = []
    for slot in opp_slots:
        c = _car_xyz(row, slot)
        if c is not None:
            d = _dist_to_goal(c[0], c[1], opp_goal_y)
            opp_cars.append((d, c))
    opp_cars.sort(key=lambda x: x[0])

    if len(opp_cars) < 1:
        return None

    # Build feature vector ─ flip y for orange perspective
    def transform(c):
        px, py, pz, vx, vy = c
        return np.array([px, sign * py, pz, vx, sign * vy], dtype=np.float32)

    feat = np.empty(FEATURE_DIM, dtype=np.float32)

    # Ball (6)
    feat[0] = ball_pos[0]
    feat[1] = sign * ball_pos[1]
    feat[2] = ball_pos[2]
    feat[3] = ball_vel[0]
    feat[4] = sign * ball_vel[1]
    feat[5] = ball_vel[2]

    # Own cars (3 × 5 = 15)
    for i in range(3):
        _, c = own_cars[i]
        feat[6 + i*5 : 6 + i*5 + 5] = transform(c)

    # Opponent summary (7)
    opp_vecs = [transform(c) for _, c in opp_cars]
    opp_arr = np.stack(opp_vecs)  # shape (n_opp, 5)
    centroid_px = float(np.mean(opp_arr[:, 0]))
    centroid_py = float(np.mean(opp_arr[:, 1]))
    centroid_vx = float(np.mean(opp_arr[:, 3]))
    centroid_vy = float(np.mean(opp_arr[:, 4]))

    # closest opp dist to their own goal (in flipped frame: their goal is at sign*opp_goal_y → becomes -own_goal_y direction)
    closest_opp_dist = float(opp_cars[0][0])

    # median opp dist to ball (dx, dy in transformed frame)
    ball_xy_t = np.array([feat[0], feat[1]], dtype=np.float32)
    opp_xy = opp_arr[:, :2]
    diffs = opp_xy - ball_xy_t
    med_idx = len(diffs) // 2
    median_d = diffs[np.argsort(np.linalg.norm(diffs, axis=1))[med_idx]]

    feat[21] = centroid_px
    feat[22] = centroid_py
    feat[23] = centroid_vx
    feat[24] = centroid_vy
    feat[25] = closest_opp_dist
    feat[26] = median_d[0]
    feat[27] = median_d[1]

    return feat


def process_logs(log_dir: str, subsample: int = SUBSAMPLE_EVERY, min_match_id: int = 0):
    """Extract formation features from tournament high-level CSV logs.

    Args:
        log_dir: Directory containing tournament CSV files.
        subsample: Keep every Nth row (~0.25 s coverage at default 8-tick sampling).
        min_match_id: Skip files whose match_id (parsed from filename prefix) is below
                      this value.  Pass last_pre_tournament_match_id + 1 to process only
                      the newest tournament's logs.
    """
    features_list = []
    meta_list = []

    csv_files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {log_dir}")

    # Filter by match_id encoded in filename: "{match_id})teamAvteamB.csv"
    if min_match_id > 0:
        def _file_match_id(path):
            try:
                return int(os.path.basename(path).split(")")[0])
            except (ValueError, IndexError):
                return 0
        csv_files = [p for p in csv_files if _file_match_id(p) >= min_match_id]

    print(f"Found {len(csv_files)} log files (min_match_id={min_match_id}).")

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        print(f"  Processing {fname} ...", end=" ", flush=True)
        count = 0
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row_idx in range(0, len(rows), subsample):
            row = rows[row_idx]
            match_id = int(row.get("match_id", 0))
            tick = int(row.get("tick", 0))

            for perspective in ("blue", "orange"):
                feat = build_feature(row, perspective)
                if feat is None:
                    continue
                features_list.append(feat)
                meta_list.append((match_id, tick, perspective))
            count += 1

        print(f"{count} samples × 2 perspectives")

    features = np.stack(features_list).astype(np.float32)
    meta_dtype = np.dtype([("match_id", "i4"), ("tick", "i4"), ("perspective", "U6")])
    metadata = np.array(meta_list, dtype=meta_dtype)

    return features, metadata


if __name__ == "__main__":
    print("Extracting formation features...")
    features, metadata = process_logs(LOG_DIR)
    print(f"Total datapoints: {len(features)}  |  Feature dim: {features.shape[1]}")

    feat_path = os.path.join(OUT_DIR, "features.npy")
    meta_path = os.path.join(OUT_DIR, "metadata.npy")
    np.save(feat_path, features)
    np.save(meta_path, metadata)
    print(f"Saved:\n  {feat_path}\n  {meta_path}")
