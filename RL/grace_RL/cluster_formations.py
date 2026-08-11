"""
cluster_formations.py
----------------------
Loads the feature matrix produced by formation_features.py and runs k-means
clustering to find k=30 representative formation states.

Outputs (saved to OUT_DIR):
  cluster_assignments.npy  — int32 array of shape (N,), cluster label per datapoint
  centroids.npy            — float32 array of shape (k, 28), cluster centroids
  cluster_info.json        — per-cluster summary: centroid, size, dominant profiles,
                             representative match_id/tick/perspective

Usage:
  python cluster_formations.py [--k 30] [--seed 42] [--no-normalize]
"""

import os
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.join(os.path.dirname(__file__), "run0", "out", "formation_clusters")

FEATURE_DIM = 28
DEFAULT_K   = 30
DEFAULT_SEED = 42


def load_data(out_dir: str):
    feat_path = os.path.join(out_dir, "features.npy")
    meta_path = os.path.join(out_dir, "metadata.npy")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"features.npy not found in {out_dir}. Run formation_features.py first.")
    features = np.load(feat_path)
    metadata = np.load(meta_path, allow_pickle=True)
    return features, metadata


def cluster(features: np.ndarray, k: int, seed: int, normalize: bool = True):
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
    else:
        scaler = None
        X = features

    print(f"Running k-means (k={k}, n={len(X)}) ...")
    if len(X) > 50_000:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=5, max_iter=300)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)

    labels = km.fit_predict(X)

    # De-normalise centroids back to original space for interpretability
    if normalize:
        centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    else:
        centroids_orig = km.cluster_centers_

    return labels.astype(np.int32), centroids_orig.astype(np.float32), scaler


def build_cluster_info(labels: np.ndarray, centroids: np.ndarray, metadata, features: np.ndarray) -> list:
    k = centroids.shape[0]
    info = []
    for c in range(k):
        mask = labels == c
        idxs = np.where(mask)[0]
        size = int(mask.sum())

        # Perspective breakdown
        perspectives = [str(metadata[i]["perspective"]) for i in idxs]
        blue_count  = perspectives.count("blue")
        orange_count = perspectives.count("orange")

        # Representative: datapoint closest to centroid in original space
        dists = np.linalg.norm(features[idxs] - centroids[c], axis=1)
        rep_idx = idxs[int(np.argmin(dists))]
        rep_meta = metadata[rep_idx]

        info.append({
            "cluster_id": c,
            "size": size,
            "blue_count": blue_count,
            "orange_count": orange_count,
            "centroid": centroids[c].tolist(),
            "representative": {
                "match_id": int(rep_meta["match_id"]),
                "tick":     int(rep_meta["tick"]),
                "perspective": str(rep_meta["perspective"]),
            },
        })

    # Sort by size descending
    info.sort(key=lambda x: x["size"], reverse=True)
    return info


def main():
    parser = argparse.ArgumentParser(description="Formation k-means clustering")
    parser.add_argument("--k",    type=int,  default=DEFAULT_K,    help="Number of clusters")
    parser.add_argument("--seed", type=int,  default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--no-normalize", action="store_true",     help="Skip StandardScaler normalization")
    args = parser.parse_args()

    features, metadata = load_data(OUT_DIR)
    print(f"Loaded {len(features)} datapoints, dim={features.shape[1]}")

    labels, centroids, scaler = cluster(features, k=args.k, seed=args.seed, normalize=not args.no_normalize)

    cluster_info = build_cluster_info(labels, centroids, metadata, features)

    # Save outputs
    assignments_path = os.path.join(OUT_DIR, "cluster_assignments.npy")
    centroids_path   = os.path.join(OUT_DIR, "centroids.npy")
    info_path        = os.path.join(OUT_DIR, "cluster_info.json")

    np.save(assignments_path, labels)
    np.save(centroids_path, centroids)
    with open(info_path, "w") as f:
        json.dump(cluster_info, f, indent=2)

    print(f"Saved:\n  {assignments_path}\n  {centroids_path}\n  {info_path}")
    print(f"\nTop-5 clusters by size:")
    for entry in cluster_info[:5]:
        print(f"  cluster {entry['cluster_id']:>2}: {entry['size']:>5} pts  "
              f"(blue={entry['blue_count']}, orange={entry['orange_count']})  "
              f"rep → match {entry['representative']['match_id']} tick {entry['representative']['tick']}")


if __name__ == "__main__":
    main()
