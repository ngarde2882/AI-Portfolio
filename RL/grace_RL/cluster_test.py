"""cluster_test.py
-----------------
Standalone entry point for the cluster replay phase.

Usage examples:
  # Use existing clusters from cycle 0, re-run replay only
  python cluster_test.py --cycle 0

  # Re-cluster from raw features, then replay
  python cluster_test.py --cycle 0 --recluster

  # Delete cached clusters and restart from scratch
  python cluster_test.py --cycle 0 --recluster --delete_clusters

  # Override how many states/cluster to replay (default from TrainConfig)
  python cluster_test.py --cycle 0 --cluster_states_per_cluster 1 --cluster_k 10

The script loads agent checkpoints from --agent_dir and --ac_dir, runs
run_cluster_phase(), saves updated policies, and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.state_mutators import KickoffMutator
from rlgym.rocket_league.sim import RocketSimEngine

from hotswap_hrl import ACConfig, ACProfilePolicy, HotswapACAdapter
from RL_two_team_env_debug import build_globals, TICKS_PER_SECOND
from AdvancedObs import AdvancedObs
from RL import (
    TrainConfig,
    _ensure_dir,
    load_or_init_policies,
    load_or_init_ac_adapters,
    save_all_policies,
    save_all_ac_policies,
    make_env,
    run_cluster_phase,
)
from formation_features import process_logs
from cluster_formations import cluster, build_cluster_info


def _recluster(cfg: TrainConfig, cycle_idx: int, pre_tournament_match_id: int) -> None:
    """Re-run feature extraction + k-means and save results."""
    out_dir = Path(cfg.out_dir)
    high_log_dir = out_dir / cfg.tournament_high_log_dirname / str(cycle_idx)
    cluster_out_dir = _ensure_dir(out_dir / "formation_clusters" / str(cycle_idx))

    print(f"[recluster] extracting features from {high_log_dir} (min_match_id={pre_tournament_match_id}) ...")
    features, metadata = process_logs(
        log_dir=str(high_log_dir),
        min_match_id=pre_tournament_match_id,
    )
    print(f"[recluster] {len(features)} datapoints extracted.")

    np.save(str(cluster_out_dir / "features.npy"), features)
    np.save(str(cluster_out_dir / "metadata.npy"), metadata)

    print(f"[recluster] clustering k={cfg.cluster_k} ...")
    labels, centroids, _ = cluster(features, k=cfg.cluster_k, seed=cfg.seed)
    cluster_info = build_cluster_info(labels, centroids, metadata, features)

    np.save(str(cluster_out_dir / "centroids.npy"), centroids)
    np.save(str(cluster_out_dir / "labels.npy"), labels)
    with open(str(cluster_out_dir / "cluster_info.json"), "w") as f:
        json.dump(cluster_info, f, indent=2)

    qualifying = [c for c in cluster_info if c["size"] >= cfg.cluster_min_density]
    print(f"[recluster] {len(qualifying)}/{len(cluster_info)} clusters qualify (density >= {cfg.cluster_min_density}).")
    print(f"[recluster] saved to {cluster_out_dir}")


def _load_cluster_summary(cfg: TrainConfig, cycle_idx: int) -> None:
    """Print a quick summary of existing cluster data."""
    cluster_out_dir = Path(cfg.out_dir) / "formation_clusters" / str(cycle_idx)
    info_path = cluster_out_dir / "cluster_info.json"
    feat_path = cluster_out_dir / "features.npy"

    if feat_path.exists():
        features = np.load(str(feat_path))
        print(f"  features: {features.shape}")

    if info_path.exists():
        with open(str(info_path)) as f:
            cluster_info = json.load(f)
        sizes = sorted([c["size"] for c in cluster_info], reverse=True)
        qualifying = [c for c in cluster_info if c["size"] >= cfg.cluster_min_density]
        print(f"  clusters: {len(cluster_info)}  qualifying: {len(qualifying)}")
        print(f"  sizes (top 10): {sizes[:10]}")
    else:
        print("  no cluster_info.json found — run with --recluster")


def main() -> None:
    p = argparse.ArgumentParser(description="Cluster replay phase test runner")
    p.add_argument("--cycle", type=int, default=0, help="Cycle index (maps to tournament_high_logs/{N})")
    p.add_argument("--pre_tournament_match_id", type=int, default=0,
                   help="Minimum match_id to include in feature extraction (0 = all files)")
    p.add_argument("--recluster", action="store_true", help="Re-extract features and re-run k-means before replay")
    p.add_argument("--delete_clusters", action="store_true",
                   help="Delete existing formation_clusters/{cycle} dir before reclustering")

    # TrainConfig overrides
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--agent_dir", type=str, default="agents")
    p.add_argument("--ac_dir", type=str, default="ac_agents")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cluster_k", type=int, default=None)
    p.add_argument("--cluster_min_density", type=int, default=None)
    p.add_argument("--cluster_states_per_cluster", type=int, default=None)
    p.add_argument("--cluster_cooldown_seconds", type=float, default=None)
    p.add_argument("--cluster_max_seconds", type=float, default=None)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--n_actions", type=int, default=90)

    args = p.parse_args()

    cfg = TrainConfig(
        out_dir=args.out_dir,
        agent_dir=args.agent_dir,
        ac_dir=args.ac_dir,
        device=args.device,
        seed=args.seed,
        action_repeat=args.action_repeat,
        n_actions=args.n_actions,
    )
    if args.cluster_k is not None:
        cfg.cluster_k = args.cluster_k
    if args.cluster_min_density is not None:
        cfg.cluster_min_density = args.cluster_min_density
    if args.cluster_states_per_cluster is not None:
        cfg.cluster_states_per_cluster = args.cluster_states_per_cluster
    if args.cluster_cooldown_seconds is not None:
        cfg.cluster_cooldown_seconds = args.cluster_cooldown_seconds
    if args.cluster_max_seconds is not None:
        cfg.cluster_max_seconds = args.cluster_max_seconds

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cycle_idx = args.cycle
    pre_tournament_match_id = args.pre_tournament_match_id

    out_dir = _ensure_dir(cfg.out_dir)
    agent_dir = _ensure_dir(cfg.agent_dir)
    ac_dir = _ensure_dir(cfg.ac_dir)

    cluster_out_dir = out_dir / "formation_clusters" / str(cycle_idx)

    # Optionally wipe cached clusters
    if args.delete_clusters and cluster_out_dir.exists():
        print(f"[test] deleting {cluster_out_dir} ...")
        shutil.rmtree(str(cluster_out_dir))

    # Print existing cluster summary before doing anything
    print(f"\n[test] cluster state for cycle {cycle_idx}:")
    _load_cluster_summary(cfg, cycle_idx)

    # Recluster if requested or if no clusters exist
    if args.recluster or not (cluster_out_dir / "cluster_info.json").exists():
        _recluster(cfg, cycle_idx, pre_tournament_match_id)

    # Build env globals
    GLOBAL_PROFILES, TEAM_SPECS = build_globals()
    team_names = list(TEAM_SPECS.keys())
    profile_names = list(GLOBAL_PROFILES.keys())

    engine = RocketSimEngine(rlbot_delay=True)
    action_parser = RepeatAction(LookupTableAction(), repeats=cfg.action_repeat)
    ll_obs_builder = AdvancedObs(profile_names=profile_names)

    # Probe obs_dim
    from RL import make_env
    ac_temp = {
        t: HotswapACAdapter(ACProfilePolicy(list(dict.fromkeys(TEAM_SPECS[t])), cfg=ACConfig(), device=cfg.device))
        for t in team_names
    }
    tmp_team = team_names[0]
    tmp_env = make_env(
        engine=engine,
        action_parser=action_parser,
        ll_obs_builder=ll_obs_builder,
        GLOBAL_PROFILES=GLOBAL_PROFILES,
        TEAM_SPECS=TEAM_SPECS,
        blue_team_name=tmp_team,
        orange_team_name=tmp_team,
        cfg=cfg,
        ac_by_team=ac_temp,
    )
    tmp_env.ac_blue = None
    tmp_env.ac_orange = None
    _, tmp_info = tmp_env.reset()
    obs_dim = next(iter(tmp_info["ll_obs"].values())).shape[0]

    tmp_state = tmp_env.state
    tmp_agent_ids = list(tmp_state.cars.keys())
    tmp_blue_aids = [aid for aid in tmp_agent_ids if int(tmp_state.cars[aid].team_num) == 0]
    tmp_ac_names = list(dict.fromkeys(TEAM_SPECS[tmp_team]))
    tmp_policy = ACProfilePolicy(tmp_ac_names, cfg=ACConfig(), device=cfg.device)
    tmp_team_obs = tmp_policy._build_team_obs(tmp_state, tmp_blue_aids)
    hl_obs_dim = int(tmp_team_obs.shape[0])

    print(f"[test] obs_dim={obs_dim}  hl_obs_dim={hl_obs_dim}  n_actions={cfg.n_actions}")

    ppo_players = load_or_init_policies(
        profile_names=profile_names,
        obs_dim=obs_dim,
        n_actions=cfg.n_actions,
        device=cfg.device,
        agent_dir=agent_dir,
    )

    ac_by_team = load_or_init_ac_adapters(
        team_names=team_names,
        TEAM_SPECS=TEAM_SPECS,
        device=cfg.device,
        ac_dir=ac_dir,
        hl_obs_dim=hl_obs_dim,
    )

    print(f"\n[test] running cluster phase (cycle={cycle_idx}, pre_tournament_match_id={pre_tournament_match_id}) ...")
    final_match_id = run_cluster_phase(
        cfg,
        engine=engine,
        action_parser=action_parser,
        ll_obs_builder=ll_obs_builder,
        GLOBAL_PROFILES=GLOBAL_PROFILES,
        TEAM_SPECS=TEAM_SPECS,
        ppo_players=ppo_players,
        ac_by_team=ac_by_team,
        start_match_id=0,
        pre_tournament_match_id=pre_tournament_match_id,
        cycle_idx=cycle_idx,
    )

    save_all_policies(ppo_players, agent_dir)
    save_all_ac_policies(ac_by_team, ac_dir)
    print(f"\n[test] done. final_match_id={final_match_id}  policies saved to '{agent_dir}' and '{ac_dir}'")

    # Print reward log summary
    log_path = out_dir / cfg.cluster_log_dirname / str(cycle_idx) / "cluster_rewards.csv"
    if log_path.exists():
        import csv
        rows = []
        with open(str(log_path)) as f:
            rows = list(csv.DictReader(f))
        if rows:
            goals = sum(int(r.get("goal_scored", 0)) for r in rows)
            avg_secs = np.mean([float(r.get("game_seconds", 0)) for r in rows])
            print(f"[test] replays={len(rows)}  goals={goals}  avg_game_secs={avg_secs:.1f}")


if __name__ == "__main__":
    main()
