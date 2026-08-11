"""RL.py

Training loop orchestration for the two-team Rocket League environment.

Phases
------
1) Gauntlet training
   - Uses pre-built GLOBAL_PROFILES / TEAM_SPECS and pre-loaded PPO policies.
   - Runs all ordered team-vs-team matchups (including self-vs-self and mirrors).
   - For each matchup, constructs the env ONCE and plays N games.
   - No high-level logs during the gauntlet.
   - Policies are saved AFTER the gauntlet stage (per training cycle) by main().

2) Tournament analysis (frozen agents)
   - Two independent 8-team brackets (two lists, each containing all 8 teams).
   - Random pairing per round, fixed-length best-of-5 (played as 5 games) to advance.
   - Final between the two bracket winners is best-of-7 (played as 7 games).
   - Agents are frozen (no transition storage, no PPO updates).
   - High-level logs enabled for 2/5 games in each best-of-5 and 3/7 in the final.
   - Tournament low-level log goes to tournament_logs.csv, with a blank line between tournaments.

This file assumes these are importable from your codebase:
  - RL_two_team_env.py (build_globals, EngineEnvAdapter, MatchRunner, PPOAgent, LowLevelLogger, etc.)
  - hotswap_hrl.py (ACProfilePolicy / Hotswap adapter)
  - AdvancedObs.py (low-level observation builder)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym.rocket_league.state_mutators import KickoffMutator
from rlgym.rocket_league.sim import RocketSimEngine

from hotswap_hrl import ACConfig, ACProfilePolicy, HotswapACAdapter

from RL_two_team_env_debug import (
    build_globals,
    TwoTeamAssignedHotswapRewardAdapter,
    EngineEnvAdapter,
    MatchRunner,
    PPOAgent,
    LowLevelLogger,
    RewardContributionLogger,
    TICKS_PER_SECOND,
    _hl_ball_half_reward,
)
from AdvancedObs import AdvancedObs
from reward_native_classes import StrikerCompositeReward, DefenderCompositeReward, PositioningCompositeReward
from formation_features import process_logs
from cluster_formations import cluster, build_cluster_info
from training_wheels import run_training_wheels


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Gauntlet
    games_per_matchup: int = 5
    gauntlet_repeats: int = 5

    # Reward annealing (gauntlet): scale rewards to encourage early exploration
    reward_scale_start: float = 100.0
    reward_scale_end: float = 1.0
    # Number of full gauntlet passes to anneal over (designed for ~100)
    reward_anneal_rounds: int = 100
    # Dense shaping is scaled freely; the goal component is capped at this value
    # after scaling so terminal events don't crush shaping signals during early training.
    # At scale=1 (end of annealing), goal*1=20 < 400 so the cap has no effect.
    goal_anneal_cap: float = 400.0
    # Training cycles (phase1 -> phase2 -> phase3 -> repeat)
    training_cycles: int = 1

    # Engine & sim
    blue_size: int = 3
    orange_size: int = 3
    action_repeat: int = 8

    # Policies
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_actions: int = 90
    agent_dir: str = "agents"

    ac_dir: str = "ac_agents"

    # Logging
    out_dir: str = "out"
    gauntlet_low_log_filename: str = "low_level_log.csv"
    tournament_low_log_filename: str = "tournament_logs.csv"
    tournament_high_log_dirname: str = "tournament_high_logs"
    high_sample_every_ticks: int = action_repeat

    # Cluster phase
    cluster_formations: bool = True          # run cluster replay phase after each tournament
    cluster_k: int = 30                      # k-means clusters
    cluster_min_density: int = 50            # skip clusters with fewer than this many datapoints
    cluster_states_per_cluster: int = 3      # replay states sampled from each qualifying cluster
    cluster_cooldown_seconds: float = 30.0   # HL acts as observer-only for first N in-game seconds
    cluster_max_seconds: float = 60.0        # end replay if no goal within this many in-game seconds
    cluster_log_dirname: str = "cluster_logs"

    # RNG
    seed: int = 42


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _policy_paths(agent_dir: Path, profile_name: str) -> Tuple[Path, Path]:
    net_p = agent_dir / f"{profile_name}.pt"
    opt_p = agent_dir / f"{profile_name}.opt.pt"
    return net_p, opt_p


def try_load_ppo_params(agent: PPOAgent, agent_dir: Path, profile_name: str) -> bool:
    net_p, opt_p = _policy_paths(agent_dir, profile_name)
    if not net_p.exists():
        return False

    state = torch.load(net_p, map_location=agent.device)
    agent.net.load_state_dict(state)

    if opt_p.exists():
        try:
            opt_state = torch.load(opt_p, map_location=agent.device)
            agent.opt.load_state_dict(opt_state)
        except Exception:
            # Optimizer state is optional.
            pass

    return True


def save_ppo_params(agent: PPOAgent, agent_dir: Path, profile_name: str) -> None:
    net_p, opt_p = _policy_paths(agent_dir, profile_name)
    torch.save(agent.net.state_dict(), net_p)
    torch.save(agent.opt.state_dict(), opt_p)


def load_or_init_policies(
    profile_names: List[str],
    obs_dim: int,
    n_actions: int,
    device: str,
    agent_dir: Path,
) -> Dict[str, PPOAgent]:
    ppo: Dict[str, PPOAgent] = {}
    loaded = 0
    for name in profile_names:
        a = PPOAgent(obs_size=obs_dim, n_actions=n_actions, device=device)
        if try_load_ppo_params(a, agent_dir, name):
            loaded += 1
        ppo[name] = a

    print(f"[policies] loaded {loaded}/{len(profile_names)} policies from '{agent_dir}'.")
    return ppo


def save_all_policies(ppo_players: Dict[str, PPOAgent], agent_dir: Path) -> None:
    for name, agent in ppo_players.items():
        save_ppo_params(agent, agent_dir, name)


# --- High-level AC policy persistence (one policy per team) ---

def _ac_policy_path(ac_dir: Path, team_name: str) -> Path:
    return ac_dir / f"{team_name}.ac.pt"


def try_load_ac_policy(adapter: HotswapACAdapter, ac_dir: Path, team_name: str, expected_obs_dim: int) -> bool:
    p = _ac_policy_path(ac_dir, team_name)
    if not p.exists():
        return False
    bundle = torch.load(p, map_location=adapter.policy.device)

    bundle_obs_dim = bundle.get("obs_dim", None)
    if bundle_obs_dim is None:
        return False
    if int(bundle_obs_dim) != int(expected_obs_dim):
        # Refuse to load incompatible checkpoints (prevents silent shape crashes)
        return False

    adapter.policy.load_from_state_dict(bundle)
    return True


def save_ac_policy(adapter: HotswapACAdapter, ac_dir: Path, team_name: str) -> None:
    p = _ac_policy_path(ac_dir, team_name)
    torch.save(adapter.policy.state_dict(), p)


def load_or_init_ac_adapters(
    team_names: List[str],
    TEAM_SPECS: Dict[str, List[str]],
    device: str,
    ac_dir: Path,
    hl_obs_dim: int,
) -> Dict[str, HotswapACAdapter]:
    ac_by_team: Dict[str, HotswapACAdapter] = {}
    loaded = 0
    for t in team_names:
        names = list(dict.fromkeys(TEAM_SPECS[t]))
        pol = ACProfilePolicy(names, cfg=ACConfig(), device=device)
        pol._ensure_net(int(hl_obs_dim))
        adapter = HotswapACAdapter(pol)
        if try_load_ac_policy(adapter, ac_dir, t, hl_obs_dim):
            loaded += 1
        ac_by_team[t] = adapter
    print(f"[ac] loaded {loaded}/{len(team_names)} team AC policies from '{ac_dir}'.")
    return ac_by_team


def save_all_ac_policies(ac_by_team: Dict[str, HotswapACAdapter], ac_dir: Path) -> None:
    for team, adapter in ac_by_team.items():
        save_ac_policy(adapter, ac_dir, team)


# -----------------------------------------------------------------------------
# Environment wiring
# -----------------------------------------------------------------------------


def make_ac_adapter_for_team(team_name: str, ac_by_team: Dict[str, HotswapACAdapter]) -> HotswapACAdapter:
    return ac_by_team[team_name]


def make_env(
    engine: RocketSimEngine,
    action_parser,
    ll_obs_builder,
    GLOBAL_PROFILES,
    TEAM_SPECS,
    blue_team_name: str,
    orange_team_name: str,
    cfg: TrainConfig,
    ac_by_team: Dict[str, HotswapACAdapter],
    reward_scale: float = 1.0,
) -> EngineEnvAdapter:
    reward_adapter = TwoTeamAssignedHotswapRewardAdapter(
        global_profiles=GLOBAL_PROFILES,
        team_specs=TEAM_SPECS,
        blue_team_name=blue_team_name,
        orange_team_name=orange_team_name,
    )

    ac_blue = make_ac_adapter_for_team(blue_team_name, ac_by_team)
    ac_orange = make_ac_adapter_for_team(orange_team_name, ac_by_team)

    env = EngineEnvAdapter(
        engine=engine,
        action_parser=action_parser,
        reward_function=reward_adapter,
        ll_obs_builder=ll_obs_builder,
        blue_size=cfg.blue_size,
        orange_size=cfg.orange_size,
        blue_team_name=blue_team_name,
        orange_team_name=orange_team_name,
        team_specs=TEAM_SPECS,
        global_profiles=GLOBAL_PROFILES,
        ac_adapter_blue=ac_blue,
        ac_adapter_orange=ac_orange,
        reward_scale=reward_scale,
        goal_anneal_cap=cfg.goal_anneal_cap,
    )
    return env


# -----------------------------------------------------------------------------
# Gauntlet
# -----------------------------------------------------------------------------


def build_all_matchups(team_names: List[str]) -> List[Tuple[str, str]]:
    """All ordered pairs (A,B) including A==B (mirrors included)."""
    return [(a, b) for a in team_names for b in team_names]


def run_gauntlet(
    cfg: TrainConfig,
    *,
    engine: RocketSimEngine,
    action_parser,
    ll_obs_builder,
    kickoff: KickoffMutator,
    GLOBAL_PROFILES,
    TEAM_SPECS,
    ppo_players: Dict[str, PPOAgent],
    ac_by_team: Dict[str, HotswapACAdapter],
    low_logger: LowLevelLogger,
    reward_logger: RewardContributionLogger,
    start_match_id: int,
) -> int:
    """Bulk training. No initialization beyond env/runner creation per matchup."""

    team_names = list(TEAM_SPECS.keys())
    base_matchups = build_all_matchups(team_names)

    match_id = start_match_id

    total_games = len(base_matchups) * cfg.games_per_matchup * cfg.gauntlet_repeats
    pbar = tqdm(total=total_games, desc="gauntlet games", dynamic_ncols=True)

    for g in range(cfg.gauntlet_repeats):
        # Reward annealing: exponential decay from scale_start to scale_end over reward_anneal_rounds.
        # Formula: scale = start * (end/start)^t  where t in [0,1].
        # With start=100, end=1: scale = 100^(1-t) → hits 10x at t=0.5 (pass 50/100).
        # This keeps the amplification high during locomotion learning, then drops quickly
        # once agents are moving, reaching the contact-quality threshold (~10x) at the midpoint.
        if cfg.reward_anneal_rounds <= 1:
            reward_scale = float(cfg.reward_scale_end)
        else:
            t = min(1.0, float(g) / float(cfg.reward_anneal_rounds - 1))
            reward_scale = float(cfg.reward_scale_start) * (float(cfg.reward_scale_end) / float(cfg.reward_scale_start)) ** t
        matchups = list(base_matchups)
        random.shuffle(matchups)

        matchup_bar = tqdm(matchups, desc=f"gauntlet matchups (pass {g+1}/{cfg.gauntlet_repeats})", leave=False, dynamic_ncols=True)
        for team_a, team_b in matchup_bar:
            blue_team, orange_team = team_a, team_b  # no coin flip; mirrors already cover both sides

            env = make_env(
                engine=engine,
                action_parser=action_parser,
                ll_obs_builder=ll_obs_builder,
                GLOBAL_PROFILES=GLOBAL_PROFILES,
                TEAM_SPECS=TEAM_SPECS,
                blue_team_name=blue_team,
                orange_team_name=orange_team,
                cfg=cfg,
                ac_by_team=ac_by_team,
                reward_scale=reward_scale,
            )
            runner = MatchRunner(env, ppo_players, kickoffs=kickoff)

            for _ in range(cfg.games_per_matchup):
                match_id += 1
                runner.run(
                    match_id=match_id,
                    low_logger=low_logger,
                    high_dir=None,  # never during gauntlet
                    reward_logger=reward_logger,
                )

                pbar.update(1)
                if (match_id % 10) == 0:
                    pbar.set_postfix_str(f"match_id={match_id} {blue_team} vs {orange_team}")

    pbar.close()
    return match_id


# -----------------------------------------------------------------------------
# Tournament (frozen analysis)
# -----------------------------------------------------------------------------


class _DummyBuffer:
    def full(self) -> bool:
        return False


class FrozenPPOAgent:
    """Read-only wrapper around PPOAgent: acts normally, does not store transitions or update."""

    def __init__(self, base: PPOAgent):
        self._base = base
        self.device = base.device
        self.net = base.net
        self.buffer = _DummyBuffer()

    def act(self, obs_np: np.ndarray):
        return self._base.act(obs_np)

    def store(self, *args, **kwargs):
        return None

    def update(self):
        return None


def _append_blank_line(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        f.write("\n")


def _run_fixed_series(
    runner: MatchRunner,
    match_id: int,
    series_len: int,
    low_logger: LowLevelLogger,
    high_dir: Path,
    high_games_to_log: int,
    high_sample_every_ticks: int,
    desc: str,
) -> Tuple[int, Dict[str, int]]:
    wins = {"BLUE": 0, "ORANGE": 0}

    for g in tqdm(range(series_len), desc=desc, leave=False, dynamic_ncols=True):
        match_id += 1
        write_high = g < high_games_to_log

        result = runner.run(
            match_id=match_id,
            low_logger=low_logger,
            high_dir=str(high_dir) if write_high else None,
            sample_every_ticks=high_sample_every_ticks,
        )
        scores = result.get("scores", {})

        if scores.get("BLUE", 0) > scores.get("ORANGE", 0):
            wins["BLUE"] += 1
        elif scores.get("ORANGE", 0) > scores.get("BLUE", 0):
            wins["ORANGE"] += 1

    return match_id, wins


def run_tournament(
    cfg: TrainConfig,
    *,
    ppo_players: Dict[str, PPOAgent],
    ac_by_team: Dict[str, HotswapACAdapter],
    GLOBAL_PROFILES,
    TEAM_SPECS,
    start_match_id: int,
    tournament_index: int = 1,
) -> int:
    """Run the tournament analysis and return the final match_id."""

    out_dir = _ensure_dir(cfg.out_dir)
    high_dir = _ensure_dir(out_dir / cfg.tournament_high_log_dirname / str(tournament_index - 1))

    # Tournament low-level logs (with blank line separation between tournaments)
    tlog_path = out_dir / cfg.tournament_low_log_filename
    _append_blank_line(tlog_path)

    profile_names = list(GLOBAL_PROFILES.keys())
    tlogger = LowLevelLogger(str(tlog_path), profile_names=profile_names)

    # Freeze low-level agents
    frozen = {name: FrozenPPOAgent(agent) for name, agent in ppo_players.items()}

    # Freeze high-level AC policies during tournament for cleaner analysis
    for _t, _ad in ac_by_team.items():
        _ad.set_training(False)

    engine = RocketSimEngine(rlbot_delay=True)
    action_parser = RepeatAction(LookupTableAction(), repeats=cfg.action_repeat)
    ll_obs_builder = AdvancedObs(profile_names=profile_names)
    kickoff = KickoffMutator()

    def play_series(team_a: str, team_b: str, series_len: int, high_k: int, match_id: int, label: str) -> Tuple[int, str]:
        # For tournament we still coin-flip sides ONCE per series for variance.
        if random.random() < 0.5:
            blue, orange = team_a, team_b
        else:
            blue, orange = team_b, team_a

        env = make_env(
            engine=engine,
            action_parser=action_parser,
            ll_obs_builder=ll_obs_builder,
            GLOBAL_PROFILES=GLOBAL_PROFILES,
            TEAM_SPECS=TEAM_SPECS,
            blue_team_name=blue,
            orange_team_name=orange,
            cfg=cfg,
            ac_by_team=ac_by_team,
                reward_scale=1.0,
        )
        runner = MatchRunner(env, frozen, kickoffs=kickoff)

        match_id, wins = _run_fixed_series(
            runner=runner,
            match_id=match_id,
            series_len=series_len,
            low_logger=tlogger,
            high_dir=high_dir,
            high_games_to_log=high_k,
            high_sample_every_ticks=cfg.high_sample_every_ticks,
            desc=label,
        )

        winner = blue if wins["BLUE"] >= wins["ORANGE"] else orange
        return match_id, winner

    def run_bracket(label: str, match_id: int) -> Tuple[int, str]:
        teams = list(TEAM_SPECS.keys())  # all 8 teams

        round_idx = 0
        while len(teams) > 1:
            round_idx += 1
            random.shuffle(teams)

            pairs = [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]
            round_bar = tqdm(pairs, desc=f"tournament {label} round {round_idx}", leave=False, dynamic_ncols=True)

            nxt: List[str] = []
            for a, b in round_bar:
                match_id, w = play_series(a, b, series_len=5, high_k=2, match_id=match_id, label=f"{label} {a} vs {b} (bo5)")
                nxt.append(w)
                round_bar.set_postfix_str(f"winner={w}")

            teams = nxt

        return match_id, teams[0]

    match_id = start_match_id

    print(f"[tournament] start idx={tournament_index} match_id={start_match_id}")
    match_id, winner_left = run_bracket("left", match_id=match_id)
    match_id, winner_right = run_bracket("right", match_id=match_id)

    print(f"[tournament] final: {winner_left} vs {winner_right}")
    match_id, champion = play_series(
        winner_left,
        winner_right,
        series_len=7,
        high_k=3,
        match_id=match_id,
        label=f"final {winner_left} vs {winner_right} (bo7)",
    )
    print(f"[tournament] champion: {champion}")

    tlogger.close()

    for _t, _ad in ac_by_team.items():
        _ad.set_training(True)

    return match_id


# -----------------------------------------------------------------------------
# Cluster phase helpers
# -----------------------------------------------------------------------------

def _new_blank_car(team_num: int) -> Car:
    """Create a Car with all fields initialized, physics zeroed."""
    from rlgym.rocket_league.common_values import OCTANE
    car = Car()
    car.team_num = team_num
    car.hitbox_type = OCTANE
    car.ball_touches = 0
    car.bump_victim_id = None
    car.demo_respawn_timer = 0.0
    car.wheels_with_contact = (True, True, True, True)
    car.supersonic_time = 0.0
    car.boost_amount = 33.0          # default 1/3 tank
    car.boost_active_time = 0.0
    car.handbrake = 0.0
    car.is_jumping = False
    car.has_jumped = False
    car.is_holding_jump = False
    car.jump_time = 0.0
    car.has_flipped = False
    car.has_double_jumped = False
    car.air_time_since_jump = 0.0
    car.flip_time = 0.0
    car.flip_torque = np.zeros(3, dtype=np.float32)
    car.is_autoflipping = False
    car.autoflip_timer = 0.0
    car.autoflip_direction = 0.0
    car._inverted_physics = None
    phys = PhysicsObject()
    phys.position = np.zeros(3, dtype=np.float32)
    phys.linear_velocity = np.zeros(3, dtype=np.float32)
    phys.angular_velocity = np.zeros(3, dtype=np.float32)
    phys._quaternion = None
    phys._euler_angles = None
    phys.rotation_mtx = np.eye(3, dtype=np.float32)
    car.physics = phys
    return car


def _rot_from_fwd_up(fwd: np.ndarray, up: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix (columns = forward, right, up) from fwd and up vectors."""
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(up, fwd)
    if np.linalg.norm(right) < 1e-6:
        # degenerate: fwd ≈ up; pick any perpendicular
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = alt - fwd * float(np.dot(alt, fwd))
    right = right / (np.linalg.norm(right) + 1e-8)
    up_out = np.cross(fwd, right)
    up_out = up_out / (np.linalg.norm(up_out) + 1e-8)
    return np.stack([fwd, right, up_out], axis=1).astype(np.float32)


def _gamestate_from_row(row: dict) -> GameState:
    """Reconstruct a GameState from a HighLevelMatchLogger CSV row."""
    gs = GameState()
    gs.tick_count = int(row.get("tick", 0))
    gs.goal_scored = False

    cfg_obj = GameConfig()
    cfg_obj.gravity = 1.0
    cfg_obj.boost_consumption = 1.0
    cfg_obj.dodge_deadzone = 0.5
    gs.config = cfg_obj

    ball = PhysicsObject()
    ball.position = np.array([float(row["ball_px"]), float(row["ball_py"]), float(row["ball_pz"])], dtype=np.float32)
    ball.linear_velocity = np.array([float(row["ball_vx"]), float(row["ball_vy"]), float(row["ball_vz"])], dtype=np.float32)
    ball.angular_velocity = np.array([
        float(row.get("ball_avx", 0) or 0),
        float(row.get("ball_avy", 0) or 0),
        float(row.get("ball_avz", 0) or 0),
    ], dtype=np.float32)
    ball._quaternion = None
    ball._euler_angles = None
    ball._rotation_mtx = None
    gs.ball = ball
    gs._inverted_ball = None

    # 34 boost pads in standard Rocket League
    gs.boost_pad_timers = np.zeros(34, dtype=np.float32)
    gs._inverted_boost_pad_timers = None

    gs.cars = {}
    for side, team_num in (("blue", BLUE_TEAM), ("orange", ORANGE_TEAM)):
        for i in range(3):
            pfx = f"{side}-{i}"
            px_val = row.get(f"{pfx}_px", "")
            if px_val == "" or px_val is None:
                continue
            car = _new_blank_car(team_num)
            p = car.physics
            p.position = np.array([float(row[f"{pfx}_px"]), float(row[f"{pfx}_py"]), float(row[f"{pfx}_pz"])], dtype=np.float32)
            p.linear_velocity = np.array([float(row[f"{pfx}_vx"]), float(row[f"{pfx}_vy"]), float(row[f"{pfx}_vz"])], dtype=np.float32)
            p.angular_velocity = np.array([
                float(row.get(f"{pfx}_avx", 0) or 0),
                float(row.get(f"{pfx}_avy", 0) or 0),
                float(row.get(f"{pfx}_avz", 0) or 0),
            ], dtype=np.float32)
            fwd = np.array([float(row[f"{pfx}_fx"]), float(row[f"{pfx}_fy"]), float(row[f"{pfx}_fz"])], dtype=np.float32)
            up  = np.array([float(row[f"{pfx}_ux"]), float(row[f"{pfx}_uy"]), float(row[f"{pfx}_uz"])], dtype=np.float32)
            p.rotation_mtx = _rot_from_fwd_up(fwd, up)
            demoed = int(row.get(f"{pfx}_demoed", 0) or 0)
            car.demo_respawn_timer = float(row.get(f"{pfx}_demo_timer", 0.0) or 0.0) if demoed else 0.0
            gs.cars[pfx] = car

    return gs


def _load_csv_rows_by_match(log_dir: str, min_match_id: int) -> dict:
    """Return {match_id: [rows]} for all tournament files with match_id >= min_match_id."""
    import glob as _g
    rows_by_match: dict = {}
    for fpath in sorted(_g.glob(str(Path(log_dir) / "*.csv"))):
        try:
            file_mid = int(Path(fpath).name.split(")")[0])
        except (ValueError, IndexError):
            file_mid = 0
        if file_mid < min_match_id:
            continue
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = int(row.get("match_id", 0))
                rows_by_match.setdefault(mid, []).append(row)
    return rows_by_match


def _load_match_teams(log_dir: str, min_match_id: int) -> Dict[int, Tuple[str, str]]:
    """Parse high-log filenames to recover (blue_team, orange_team) per match_id.

    Filenames use the format: {match_id}){blue_team}v{orange_team}.csv
    """
    import glob as _g
    teams: Dict[int, Tuple[str, str]] = {}
    for fpath in sorted(_g.glob(str(Path(log_dir) / "*.csv"))):
        stem = Path(fpath).stem
        try:
            mid = int(stem.split(")")[0])
        except (ValueError, IndexError):
            continue
        if mid < min_match_id:
            continue
        rest = stem.split(")", 1)[1] if ")" in stem else ""
        if "__v__" in rest:
            parts = rest.split("__v__", 1)
            teams[mid] = (parts[0], parts[1])
    return teams


def _profile_rotation(profiles: List[str], n_runs: int = 3) -> List[List[str]]:
    """
    Return n_runs assignment lists of length 3 (one per car slot).
    Each unique profile appears in each slot position at least once across runs.
    Uses the 3× pool strategy: triple the unique profile list, shuffle, then slice.
    Used only for one-off defender initial assignments.
    """
    unique = list(dict.fromkeys(profiles))
    needed = n_runs * 3
    pool = (unique * (needed // len(unique) + 1))[:needed]  # enough copies to fill all slots
    random.shuffle(pool)
    return [pool[i * 3:(i + 1) * 3] for i in range(n_runs)]


def _slot_assignments_epoch(profiles: List[str]) -> List[List[str]]:
    """
    One epoch of 3-slot assignments covering every unique profile exactly once per slot.

    Each slot gets its own independent shuffle of unique profiles, then assignments
    are built by popping one profile from each slot list per round.

    Returns len(unique) assignments, each a list of 3 profiles (one per car slot).
    For a single-profile team like ["s0"] this returns exactly [[s0, s0, s0]].
    """
    unique = list(dict.fromkeys(profiles))
    slot_lists = [random.sample(unique, len(unique)) for _ in range(3)]
    return [[slot_lists[pos][i] for pos in range(3)] for i in range(len(unique))]


# -----------------------------------------------------------------------------
# Cluster phase runner
# -----------------------------------------------------------------------------

_HL_TOUCH_REWARD = 0.02  # HL reward pulse when any team player touches the ball

def run_cluster_phase(
    cfg: TrainConfig,
    *,
    engine: RocketSimEngine,
    action_parser,
    ll_obs_builder,
    GLOBAL_PROFILES,
    TEAM_SPECS,
    ppo_players: Dict[str, PPOAgent],
    ac_by_team: Dict[str, HotswapACAdapter],
    start_match_id: int,
    pre_tournament_match_id: int,
    cycle_idx: int,
) -> int:
    """
    Phase 3: cluster replay training.

    1. Feature-extract + cluster the new tournament high-level logs.
    2. For each qualifying cluster, sample states and build a replay pool.
    3. For each (state, team_pair, slot_rotation) run a capped replay:
       - First cluster_cooldown_seconds: HL observes only (profiles locked).
       - After cooldown: HL decides profiles normally.
       - Ends on goal or cluster_max_seconds in-game time.
    4. Finalize HL training segments and log reward breakdowns.
    """
    out_dir = _ensure_dir(cfg.out_dir)
    high_log_dir = out_dir / cfg.tournament_high_log_dirname / str(cycle_idx)
    cluster_out_dir = _ensure_dir(out_dir / "formation_clusters" / str(cycle_idx))
    cluster_log_dir = _ensure_dir(out_dir / cfg.cluster_log_dirname / str(cycle_idx))

    # ── 1. Feature extraction ──────────────────────────────────────────────
    print(f"[cluster] extracting features from match_id >= {pre_tournament_match_id} ...")
    try:
        features, metadata = process_logs(
            log_dir=str(high_log_dir),
            min_match_id=pre_tournament_match_id,
        )
    except FileNotFoundError as e:
        print(f"[cluster] no tournament logs found: {e}  — skipping cluster phase.")
        return start_match_id

    if len(features) == 0:
        print("[cluster] no usable formation features — skipping cluster phase.")
        return start_match_id

    np.save(str(cluster_out_dir / "features.npy"), features)
    np.save(str(cluster_out_dir / "metadata.npy"), metadata)
    print(f"[cluster] {len(features)} formation datapoints extracted.")

    # ── 2. Clustering ─────────────────────────────────────────────────────
    print(f"[cluster] clustering k={cfg.cluster_k} ...")
    labels, centroids, _ = cluster(features, k=cfg.cluster_k, seed=cfg.seed)
    cluster_info = build_cluster_info(labels, centroids, metadata, features)
    np.save(str(cluster_out_dir / "centroids.npy"), centroids)
    with open(str(cluster_out_dir / "cluster_info.json"), "w") as fj:
        json.dump(cluster_info, fj, indent=2)

    qualifying = [c for c in cluster_info if c["size"] >= cfg.cluster_min_density]
    print(f"[cluster] {len(qualifying)}/{len(cluster_info)} clusters qualify (density >= {cfg.cluster_min_density}).")

    if not qualifying:
        print("[cluster] no qualifying clusters — skipping replay phase.")
        return start_match_id

    # ── 3. Load CSV rows for state lookup ──────────────────────────────────
    rows_by_match = _load_csv_rows_by_match(str(high_log_dir), pre_tournament_match_id)

    # Build a lookup: (match_id, tick) → row
    tick_lookup: dict = {}
    for mid, rows in rows_by_match.items():
        for row in rows:
            tick_lookup[(mid, int(row.get("tick", 0)))] = row

    # Parse original (blue_team, orange_team) per match from filenames
    match_teams: Dict[int, Tuple[str, str]] = _load_match_teams(str(high_log_dir), pre_tournament_match_id)

    team_names = list(TEAM_SPECS.keys())

    # Cluster phase reward log
    log_path = str(cluster_log_dir / "cluster_rewards.csv")
    log_fields = [
        "match_id", "cluster_id", "replay_idx", "perspective",
        "owner_team", "defender_team", "blue_team", "orange_team",
        "ticks_played", "game_seconds", "goal_scored", "scoring_team",
        "ac_tick_blue", "ac_tick_orange",
    ]
    reward_keys_all: List[str] = sorted(set(
        list(StrikerCompositeReward().default_weights().keys()) +
        list(DefenderCompositeReward().default_weights().keys()) +
        list(PositioningCompositeReward().default_weights().keys()) +
        ["cost_of_living"]
    ))
    for k in reward_keys_all:
        log_fields += [f"blue_{k}", f"orange_{k}"]

    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    match_id = start_match_id
    cooldown_ticks = int(cfg.cluster_cooldown_seconds * TICKS_PER_SECOND / cfg.action_repeat)
    max_ticks = int(cfg.cluster_max_seconds * TICKS_PER_SECOND / cfg.action_repeat)

    pbar = tqdm(qualifying, desc="cluster replay clusters", dynamic_ncols=True)

    for cluster_entry in pbar:
        cid = cluster_entry["cluster_id"]
        cluster_size = cluster_entry["size"]
        cluster_idxs = np.where(labels == cid)[0]

        n_sample = min(cfg.cluster_states_per_cluster, len(cluster_idxs))
        sampled_idxs = np.random.choice(cluster_idxs, size=n_sample, replace=False)

        for s_idx in sampled_idxs:
            rep_meta = metadata[s_idx]
            mid = int(rep_meta["match_id"])
            tick = int(rep_meta["tick"])
            perspective = str(rep_meta["perspective"])  # "blue" or "orange"
            row = tick_lookup.get((mid, tick))
            if row is None:
                continue

            gs = _gamestate_from_row(row)
            if len(gs.cars) < 6:
                continue

            # Recover which teams played this match from the log filename.
            # owner_team is the team whose formation this state represents.
            # original_defender is the opponent in that match.
            orig_blue, orig_orange = match_teams.get(mid, (team_names[0], team_names[0]))
            owner_team = orig_blue if perspective == "blue" else orig_orange
            original_defender = orig_orange if perspective == "blue" else orig_blue

            # Owner always plays on their original side so the formation is preserved.
            attacker_is_blue = (perspective == "blue")

            owner_profiles = list(TEAM_SPECS[owner_team])

            # 10 defender configs: all 8 teams once + original_defender twice extra.
            # This weights the original opponent but exposes the state to every team.
            defender_pool = list(team_names) + [original_defender, original_defender]
            random.shuffle(defender_pool)

            replay_idx = 0
            for defender_team in defender_pool:  # 10 defender iterations
                defender_profiles = list(TEAM_SPECS[defender_team])
                # Defender gets one initial slot assignment; HL takes over immediately.
                def_assign = _profile_rotation(defender_profiles, n_runs=1)[0]

                blue_team   = owner_team    if attacker_is_blue else defender_team
                orange_team = defender_team if attacker_is_blue else owner_team

                # Build env once per defender; reset for each attacker slot assignment.
                env = make_env(
                    engine=engine,
                    action_parser=action_parser,
                    ll_obs_builder=ll_obs_builder,
                    GLOBAL_PROFILES=GLOBAL_PROFILES,
                    TEAM_SPECS=TEAM_SPECS,
                    blue_team_name=blue_team,
                    orange_team_name=orange_team,
                    cfg=cfg,
                    ac_by_team=ac_by_team,
                    reward_scale=1.0,
                )

                for att_assign in _slot_assignments_epoch(owner_profiles):
                    match_id += 1

                    b_assign = att_assign if attacker_is_blue else def_assign
                    o_assign = def_assign if attacker_is_blue else att_assign

                    _, info = env.reset(initial_state=gs)
                    agent_ids = list(env.state.cars.keys())
                    blue_aids   = [a for a in agent_ids if int(env.state.cars[a].team_num) == BLUE_TEAM]
                    orange_aids = [a for a in agent_ids if int(env.state.cars[a].team_num) == ORANGE_TEAM]

                    # Apply initial slot assignments (HL overrides immediately for defender, after cooldown for attacker)
                    for i, aid in enumerate(sorted(blue_aids)):
                        prof = b_assign[i % len(b_assign)]
                        env.player_by_agent[aid] = prof
                        env.reward_function.set_profile(aid, env.state, prof)
                    for i, aid in enumerate(sorted(orange_aids)):
                        prof = o_assign[i % len(o_assign)]
                        env.player_by_agent[aid] = prof
                        env.reward_function.set_profile(aid, env.state, prof)

                    ac_blue_adapter:   HotswapACAdapter = ac_by_team[blue_team]
                    ac_orange_adapter: HotswapACAdapter = ac_by_team[orange_team]
                    ac_blue_adapter.policy.reset()
                    ac_orange_adapter.policy.reset()

                    # Attacker AC: observe-only during cooldown.
                    # Defender AC: free to act from tick 0 (no cooldown).
                    attacker_ac   = ac_blue_adapter  if attacker_is_blue else ac_orange_adapter
                    attacker_aids = blue_aids        if attacker_is_blue else orange_aids

                    ac_tick_total_blue  = 0.0
                    ac_tick_total_orange = 0.0
                    team_reward_sums: Dict[int, dict] = {0: {}, 1: {}}
                    tick_counter = 0
                    goal_scored = False
                    scoring_team_str = ""

                    while tick_counter < max_ticks:
                        actions: Dict = {}
                        actions_idx: Dict = {}
                        cur_profiles = dict(info["profile_by_agent"])

                        prev_obs:  Dict = {}
                        prev_logp: Dict = {}
                        prev_val:  Dict = {}
                        for aid in agent_ids:
                            pname = cur_profiles.get(aid, list(GLOBAL_PROFILES.keys())[0])
                            a, logp, val = ppo_players[pname].act(info["ll_obs"][aid])
                            actions[aid]     = np.array([a], dtype=np.int64)
                            actions_idx[aid] = int(a)
                            prev_obs[aid]    = info["ll_obs"][aid]
                            prev_logp[aid]   = float(logp)
                            prev_val[aid]    = float(val)

                        in_cooldown = tick_counter < cooldown_ticks

                        if in_cooldown:
                            # Null attacker AC only so step() won't switch attacker profiles.
                            # Defender AC remains on env so it can decide freely.
                            if attacker_is_blue:
                                env.ac_blue = None
                            else:
                                env.ac_orange = None

                        _, _, done, info = env.step(actions, shared_info={"actions_idx": actions_idx})

                        if in_cooldown:
                            # Restore attacker AC and feed observation to buffer without acting.
                            if attacker_is_blue:
                                env.ac_blue = attacker_ac
                            else:
                                env.ac_orange = attacker_ac
                            attacker_ac.observe_only(env.state, attacker_aids)
                        # Defender AC: step() already called decide_and_update every tick.

                        state_after = env.state
                        tick_counter += 1

                        # Tick rewards: ball-half position + touch bonus
                        r_tick_b = _hl_ball_half_reward(state_after, team=0)
                        r_tick_o = _hl_ball_half_reward(state_after, team=1)
                        last_touch = info.get("last_touch", {})
                        if isinstance(last_touch, dict) and last_touch.get("tick") == int(state_after.tick_count):
                            touch_team = last_touch.get("team")
                            if touch_team == 0:
                                r_tick_b += _HL_TOUCH_REWARD
                            elif touch_team == 1:
                                r_tick_o += _HL_TOUCH_REWARD
                        ac_tick_total_blue   += r_tick_b
                        ac_tick_total_orange += r_tick_o
                        ac_blue_adapter.add_tick_reward(r_tick_b)
                        ac_orange_adapter.add_tick_reward(r_tick_o)

                        # Accumulate reward breakdown
                        rb = info.get("reward_breakdown", {})
                        for aid in agent_ids:
                            t = int(state_after.cars[aid].team_num)
                            for k, v_r in rb.get(aid, {}).items():
                                team_reward_sums[t][k] = float(team_reward_sums[t].get(k, 0.0)) + float(v_r)

                        # PPO store
                        rmap = dict(info["rewards"])
                        for aid in agent_ids:
                            pname_prev = cur_profiles.get(aid, list(GLOBAL_PROFILES.keys())[0])
                            ppo_players[pname_prev].store(
                                prev_obs[aid],
                                int(actions[aid][0]),
                                prev_logp[aid],
                                float(rmap.get(aid, 0.0)),
                                prev_val[aid],
                                bool(done),
                            )
                        for agent in ppo_players.values():
                            if agent.buffer.full():
                                agent.update()

                        if state_after.goal_scored:
                            goal_scored = True
                            scoring_team_str = "blue" if float(state_after.ball.position[1]) > 0 else "orange"
                            _blue_scored = scoring_team_str == "blue"
                            ac_blue_adapter.finalize_goal(+1.0 if _blue_scored else -1.0)
                            ac_orange_adapter.finalize_goal(+1.0 if not _blue_scored else -1.0)
                            break

                        if done:
                            break

                    # Flush remaining HL segment
                    ac_blue_adapter.finalize_match(0.0)
                    ac_orange_adapter.finalize_match(0.0)

                    # Write cluster log row
                    log_row: dict = {
                        "match_id": match_id,
                        "cluster_id": cid,
                        "replay_idx": replay_idx,
                        "perspective": perspective,
                        "owner_team": owner_team,
                        "defender_team": defender_team,
                        "blue_team": blue_team,
                        "orange_team": orange_team,
                        "ticks_played": tick_counter,
                        "game_seconds": round(tick_counter * cfg.action_repeat / TICKS_PER_SECOND, 2),
                        "goal_scored": int(goal_scored),
                        "scoring_team": scoring_team_str,
                        "ac_tick_blue": round(ac_tick_total_blue, 6),
                        "ac_tick_orange": round(ac_tick_total_orange, 6),
                    }
                    for k in reward_keys_all:
                        log_row[f"blue_{k}"] = round(float(team_reward_sums[0].get(k, 0.0)), 6)
                        log_row[f"orange_{k}"] = round(float(team_reward_sums[1].get(k, 0.0)), 6)
                    log_writer.writerow(log_row)
                    log_file.flush()

                    replay_idx += 1

        pbar.set_postfix_str(f"cluster={cid} size={cluster_size}")

    log_file.close()
    print(f"[cluster] phase complete. replays logged to '{log_path}'.")
    return match_id


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Two-team Rocket League gauntlet + tournament driver")

    p.add_argument("--games_per_matchup", type=int, default=4)#5
    p.add_argument("--gauntlet_repeats", type=int, default=2)#5
    p.add_argument("--training_cycles", type=int, default=2)#3
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--agent_dir", type=str, default="agents")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    return TrainConfig(
        games_per_matchup=int(args.games_per_matchup),
        gauntlet_repeats=int(args.gauntlet_repeats),
        training_cycles=int(args.training_cycles),
        out_dir=str(args.out_dir),
        agent_dir=str(args.agent_dir),
        device=str(args.device),
        seed=int(args.seed),
    )


def main() -> None:
    cfg = parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = _ensure_dir(cfg.out_dir)
    agent_dir = _ensure_dir(cfg.agent_dir)

    GLOBAL_PROFILES, TEAM_SPECS = build_globals()
    team_names = list(TEAM_SPECS.keys())
    profile_names = list(GLOBAL_PROFILES.keys())

    reward_keys = set()
    reward_keys |= set(StrikerCompositeReward().default_weights().keys())
    reward_keys |= set(DefenderCompositeReward().default_weights().keys())
    reward_keys |= set(PositioningCompositeReward().default_weights().keys())
    reward_keys.add("cost_of_living")
    reward_keys = sorted(reward_keys)

    engine = RocketSimEngine(rlbot_delay=True)
    action_parser = RepeatAction(LookupTableAction(), repeats=cfg.action_repeat)
    ll_obs_builder = AdvancedObs(profile_names=profile_names)
    kickoff = KickoffMutator()

    # Determine obs_dim via a temporary reset (AC adapters are not used during reset)
    tmp_team = team_names[0]

    # Temporary AC adapters (uninitialized nets) just to satisfy env wiring
    ac_temp = {t: HotswapACAdapter(ACProfilePolicy(list(dict.fromkeys(TEAM_SPECS[t])), cfg=ACConfig(), device=cfg.device)) for t in team_names}
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
    # Disable AC during obs-dim probing; AC nets depend on obs_dim.
    tmp_env.ac_blue = None
    tmp_env.ac_orange = None
    _, tmp_info = tmp_env.reset()
    obs_dim = next(iter(tmp_info["ll_obs"].values())).shape[0]

    # Determine high-level (AC) obs_dim via a temporary team-obs build
    tmp_state = tmp_env.state
    tmp_agent_ids = list(tmp_state.cars.keys())
    tmp_blue_aids = [aid for aid in tmp_agent_ids if int(tmp_state.cars[aid].team_num) == 0]
    tmp_ac_names = list(dict.fromkeys(TEAM_SPECS[tmp_team]))
    tmp_policy = ACProfilePolicy(tmp_ac_names, cfg=ACConfig(), device=cfg.device)
    tmp_team_obs = tmp_policy._build_team_obs(tmp_state, tmp_blue_aids)
    hl_obs_dim = int(tmp_team_obs.shape[0])

    # print(f"[env] ll_obs_dim={obs_dim}, hl_obs_dim={hl_obs_dim}, n_actions={cfg.n_actions}")

    ac_dir = _ensure_dir(cfg.ac_dir)

    fresh_start = not any(_policy_paths(agent_dir, n)[0].exists() for n in profile_names)

    ppo_players = load_or_init_policies(
        profile_names=profile_names,
        obs_dim=obs_dim,
        n_actions=cfg.n_actions,
        device=cfg.device,
        agent_dir=agent_dir,
    )

    if fresh_start:
        print("[training_wheels] No checkpoints found — running pre-training curriculum.")
        tw_log = str(out_dir / "training_wheels_log.csv")
        run_training_wheels(ppo_players, GLOBAL_PROFILES, profile_names, log_path=tw_log)
        save_all_policies(ppo_players, agent_dir)
        print(f"[training_wheels] Pre-training complete. Checkpoints saved. Log: {tw_log}")

    ac_by_team = load_or_init_ac_adapters(
        team_names=team_names,
        TEAM_SPECS=TEAM_SPECS,
        device=cfg.device,
        ac_dir=ac_dir,
        hl_obs_dim=hl_obs_dim,
    )

    # Gauntlet low-level logger can persist across cycles (single file, appended by logger behavior).
    gauntlet_low_logger = LowLevelLogger(str(out_dir / cfg.gauntlet_low_log_filename), profile_names=profile_names)
    # Reward logger running in parallel to capture reward breakdowns per timestep.
    reward_logger = RewardContributionLogger(str(out_dir / "reward_contrib.csv"), reward_keys=reward_keys)

    last_match_id = 0

    for cycle_idx in range(cfg.training_cycles):
        if cfg.training_cycles > 1:
            print(f"[cycle] {cycle_idx+1}/{cfg.training_cycles}")

        # Phase 1: Gauntlet bulk training
        last_match_id = run_gauntlet(
            cfg,
            engine=engine,
            action_parser=action_parser,
            ll_obs_builder=ll_obs_builder,
            kickoff=kickoff,
            GLOBAL_PROFILES=GLOBAL_PROFILES,
            TEAM_SPECS=TEAM_SPECS,
            ppo_players=ppo_players,
            ac_by_team=ac_by_team,
            low_logger=gauntlet_low_logger,
            reward_logger=reward_logger,
            start_match_id=last_match_id,
        )

        # Save once after gauntlet repeats for this cycle.
        save_all_policies(ppo_players, agent_dir)
        save_all_ac_policies(ac_by_team, ac_dir)
        print(f"[gauntlet] cycle {cycle_idx+1}: policies saved to '{agent_dir}' and '{ac_dir}'.")

        # Phase 2: Tournament analysis (agents frozen)
        pre_tournament_match_id = last_match_id + 1
        last_match_id = run_tournament(
            cfg,
            ppo_players=ppo_players,
            ac_by_team=ac_by_team,
            GLOBAL_PROFILES=GLOBAL_PROFILES,
            TEAM_SPECS=TEAM_SPECS,
            start_match_id=last_match_id,
            tournament_index=cycle_idx + 1,
        )

        # Phase 3: Cluster replay training
        if cfg.cluster_formations:
            last_match_id = run_cluster_phase(
                cfg,
                engine=engine,
                action_parser=action_parser,
                ll_obs_builder=ll_obs_builder,
                GLOBAL_PROFILES=GLOBAL_PROFILES,
                TEAM_SPECS=TEAM_SPECS,
                ppo_players=ppo_players,
                ac_by_team=ac_by_team,
                start_match_id=last_match_id,
                pre_tournament_match_id=pre_tournament_match_id,
                cycle_idx=cycle_idx,
            )
            save_all_policies(ppo_players, agent_dir)
            save_all_ac_policies(ac_by_team, ac_dir)
            print(f"[cluster] cycle {cycle_idx+1}: policies saved.")

    gauntlet_low_logger.close()

    # Final save (end of run)
    save_all_policies(ppo_players, agent_dir)
    save_all_ac_policies(ac_by_team, ac_dir)
    print(f"[done] complete. last_match_id={last_match_id} policies_dir='{agent_dir}' ac_policies_dir='{ac_dir}'")


if __name__ == "__main__":
    main()
