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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
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
)
from AdvancedObs import AdvancedObs
from reward_native_classes import StrikerCompositeReward, DefenderCompositeReward, PositioningCompositeReward
from RL_two_team_env_debug import RewardContributionLogger


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
    # At scale=1 (end of annealing), goal*1=10 < 400 so the cap has no effect.
    goal_anneal_cap: float = 400.0
    # Training cycles (phase1 -> phase2 -> phase3 -> repeat)
    training_cycles: int = 1

    # Engine & sim
    blue_size: int = 3
    orange_size: int = 3
    action_repeat: int = 8

    # Policies
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    n_actions: int = 90
    agent_dir: str = "agents"

    ac_dir: str = "ac_agents"

    # Logging
    out_dir: str = "out"
    gauntlet_low_log_filename: str = "low_level_log.csv"
    tournament_low_log_filename: str = "tournament_logs.csv"
    tournament_high_log_dirname: str = "tournament_high_logs"
    high_sample_every_ticks: int = action_repeat

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
        # Reward annealing: start high to amplify tiny dense shaping rewards, then decay toward 1.0
        if cfg.reward_anneal_rounds <= 1:
            reward_scale = float(cfg.reward_scale_end)
        else:
            t = min(1.0, float(g) / float(cfg.reward_anneal_rounds - 1))
            reward_scale = float(cfg.reward_scale_start + t * (cfg.reward_scale_end - cfg.reward_scale_start))
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
    high_dir = _ensure_dir(out_dir / cfg.tournament_high_log_dirname)

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
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Two-team Rocket League gauntlet + tournament driver")

    # Defaults kept; you can wire more flags later.
    p.add_argument("--games_per_matchup", type=int, default=5)
    p.add_argument("--gauntlet_repeats", type=int, default=5)
    p.add_argument("--training_cycles", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--agent_dir", type=str, default="agents")
    p.add_argument("--device", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
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
        last_match_id = run_tournament(
            cfg,
            ppo_players=ppo_players,
            ac_by_team=ac_by_team,
            GLOBAL_PROFILES=GLOBAL_PROFILES,
            TEAM_SPECS=TEAM_SPECS,
            start_match_id=last_match_id,
            tournament_index=cycle_idx + 1,
        )

        # Phase 3 hook: (future) additional training stage here.

    gauntlet_low_logger.close()

    # Final save (end of run)
    save_all_policies(ppo_players, agent_dir)
    save_all_ac_policies(ac_by_team, ac_dir)
    print(f"[done] complete. last_match_id={last_match_id} policies_dir='{agent_dir}' ac_policies_dir='{ac_dir}'")


if __name__ == "__main__":
    main()
