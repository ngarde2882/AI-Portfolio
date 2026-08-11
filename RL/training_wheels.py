"""training_wheels.py

One-shot pre-training curriculum for low-level PPO agents.
Runs automatically the first time RL.py main() is called when no agent
checkpoints exist.  Can also be run standalone:

    python training_wheels.py [--agents agents/] [--device cpu]

Design
------
Each profile trains independently using its own PPOAgent so that the
profile one-hot encoded in AdvancedObs is always correct for that agent.
A temporary env is created per profile with a team spec containing only
that profile — all 6 car slots are assigned to it, but 5 of those cars
are stationary dummies parked in the corners.

Drills (in order, same set run for every profile):
  OFF-1  observe_goal   : ball already heading to orange goal; agent far away
                          (observe_only — no actions stored). 500 reps.
  OFF-2  easy_tap       : ball ~300uu in front of goal rolling slowly; agent
                          right behind it. 1000 reps.
  OFF-3  shot_window    : ball 1500-2500uu from goal at angle; agent in
                          strike position. 1000 reps.
  DEF-1  slow_cant_reach: ball heading slowly to own goal; agent 4500uu away
                          (can't reach). 300 reps.
  DEF-2  slow_can_reach : same but agent 1000-2000uu away — can intercept.
                          800 reps.
  DEF-3  bouncy_fast    : faster ball with upward velocity toward own goal.
                          600 reps.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from rlgym.rocket_league.api import GameConfig, GameState, PhysicsObject
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, TICKS_PER_SECOND
from rlgym.rocket_league.sim import RocketSimEngine

from RL_two_team_env_debug import (
    build_globals,
    TwoTeamAssignedHotswapRewardAdapter,
    EngineEnvAdapter,
    PPOAgent,
    LowLevelLogger,
)
def _new_blank_car(team_num: int):
    from rlgym.rocket_league.api import Car, PhysicsObject
    from rlgym.rocket_league.common_values import OCTANE
    car = Car()
    car.team_num = team_num
    car.hitbox_type = OCTANE
    car.ball_touches = 0
    car.bump_victim_id = None
    car.demo_respawn_timer = 0.0
    car.wheels_with_contact = (True, True, True, True)
    car.supersonic_time = 0.0
    car.boost_amount = 33.0
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
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(up, fwd)
    if np.linalg.norm(right) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = alt - fwd * float(np.dot(alt, fwd))
    right = right / (np.linalg.norm(right) + 1e-8)
    up_out = np.cross(fwd, right)
    up_out = up_out / (np.linalg.norm(up_out) + 1e-8)
    return np.stack([fwd, right, up_out], axis=1).astype(np.float32)
from AdvancedObs import AdvancedObs
from hotswap_hrl import ACConfig, ACProfilePolicy, HotswapACAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUND_Z      = 93.15    # ball resting on ground (ball radius)
CAR_Z         = 17.0     # car centroid height on ground
MAX_BOOST     = 100.0
ACTION_REPEAT = 8

DRILL_SECONDS   = 5.0
TICKS_PER_DRILL = int(DRILL_SECONDS * TICKS_PER_SECOND / ACTION_REPEAT)

# Dummy team name used for both sides of the drill env
_DRILL_TEAM = "_tw_team"

DRILLS: List[Dict] = [
    {"name": "observe_goal",    "reps": 500,  "observe_only": True,  "defensive": False},
    {"name": "easy_tap",        "reps": 1000, "observe_only": False, "defensive": False},
    {"name": "shot_window",     "reps": 1000, "observe_only": False, "defensive": False},
    {"name": "slow_cant_reach", "reps": 300,  "observe_only": False, "defensive": True},
    {"name": "slow_can_reach",  "reps": 800,  "observe_only": False, "defensive": True},
    {"name": "bouncy_fast",     "reps": 600,  "observe_only": False, "defensive": True},
]

NULL_ACTION_IDX = 0

# ---------------------------------------------------------------------------
# GameState builders
# ---------------------------------------------------------------------------

def _base_gs() -> GameState:
    gs = GameState()
    gs.tick_count = 0
    gs.goal_scored = False
    cfg = GameConfig()
    cfg.gravity = 1.0
    cfg.boost_consumption = 1.0
    cfg.dodge_deadzone = 0.5
    gs.config = cfg
    gs.boost_pad_timers = np.zeros(34, dtype=np.float32)
    gs._inverted_boost_pad_timers = None
    gs._inverted_ball = None
    gs.cars = {}
    return gs


def _set_ball(gs: GameState, pos, vel, ang_vel=None):
    ball = PhysicsObject()
    ball.position = np.asarray(pos, dtype=np.float32)
    ball.linear_velocity = np.asarray(vel, dtype=np.float32)
    ball.angular_velocity = np.zeros(3, dtype=np.float32) if ang_vel is None else np.asarray(ang_vel, dtype=np.float32)
    ball._quaternion = None
    ball._euler_angles = None
    ball._rotation_mtx = None
    gs.ball = ball


def _add_car(gs: GameState, aid: str, team: int, pos, vel, fwd, boost: float = 33.0):
    car = _new_blank_car(team)
    car.boost_amount = float(boost)
    p = car.physics
    p.position = np.asarray(pos, dtype=np.float32)
    p.linear_velocity = np.asarray(vel, dtype=np.float32)
    p.angular_velocity = np.zeros(3, dtype=np.float32)
    p.rotation_mtx = _rot_from_fwd_up(np.asarray(fwd, dtype=np.float32),
                                       np.array([0., 0., 1.], dtype=np.float32))
    gs.cars[aid] = car


def _fill_dummies(gs: GameState, train_aid: str):
    """Park the 5 non-training cars in corners, far from play."""
    corners = [
        [-3500., -4500., CAR_Z], [ 3500., -4500., CAR_Z],
        [-3500.,  4500., CAR_Z], [ 3500.,  4500., CAR_Z],
        [    0., -4800., CAR_Z],
    ]
    all_slots = ["blue-0", "blue-1", "blue-2", "orange-0", "orange-1", "orange-2"]
    dummies = [s for s in all_slots if s != train_aid]
    for i, slot in enumerate(dummies):
        team = BLUE_TEAM if slot.startswith("blue") else ORANGE_TEAM
        _add_car(gs, slot, team, corners[i], [0., 0., 0.], [0., 1., 0.], boost=0.0)


# ── Drill state builders ────────────────────────────────────────────────────

def build_observe_goal() -> GameState:
    """Ball in the air heading toward orange goal; agent parked far away (observe only).

    Ball starts at y∈[3000,4500] elevated ~300uu so ground friction is minimal, with
    vy∈[1500,2500] — guaranteed to reach the goal in well under 5s.
    """
    gs = _base_gs()
    bx = random.uniform(-400., 400.)
    by = random.uniform(3000., 4500.)   # close enough to goal that friction is irrelevant
    bz = random.uniform(200., 400.)     # in the air — no ground friction drag
    vy = random.uniform(1500., 2500.)   # fast enough to guarantee arrival
    _set_ball(gs, [bx, by, bz], [random.uniform(-80., 80.), vy, random.uniform(-50., 50.)])
    _add_car(gs, "blue-0", BLUE_TEAM,
             [random.uniform(-1000., 1000.), random.uniform(-4000., -2500.), CAR_Z],
             [0., 0., 0.], [0., 1., 0.], boost=MAX_BOOST)
    _fill_dummies(gs, "blue-0")
    return gs


def build_easy_tap() -> GameState:
    """Agent starts already touching the ball just outside the orange goal.

    Agent is placed 50-120uu behind the ball with initial speed 600-900 uu/s so
    contact happens within the first few ticks regardless of network output.
    Ball is elevated slightly so it doesn't ground-friction-stop before the goal line.
    """
    gs = _base_gs()
    bx = random.uniform(-500., 500.)
    by = random.uniform(4400., 4900.)   # 220-720uu from goal
    bz = random.uniform(GROUND_Z, 150.) # slight elevation reduces friction
    _set_ball(gs, [bx, by, bz], [random.uniform(-30., 30.), random.uniform(50., 150.), 0.])
    # Agent starts right behind the ball — contact on tick 1
    ay = by - random.uniform(50., 120.)
    asp = random.uniform(600., 900.)
    _add_car(gs, "blue-0", BLUE_TEAM,
             [bx + random.uniform(-40., 40.), ay, CAR_Z],
             [0., asp, 0.], [0., 1., 0.], boost=MAX_BOOST)
    _fill_dummies(gs, "blue-0")
    return gs


def build_shot_window() -> GameState:
    """Ball 1500-2500uu from orange goal at angle; agent intercepting."""
    gs = _base_gs()
    bx = random.uniform(-1500., 1500.)
    by = random.uniform(2000., 3500.)
    _set_ball(gs, [bx, by, GROUND_Z],
              [random.uniform(-200., 200.), random.uniform(300., 700.), 0.])
    ax = bx + random.uniform(-500., 500.)
    ay = by - random.uniform(400., 900.)
    spd = random.uniform(400., 900.)
    dx, dy = bx - ax, by - ay
    norm = max(np.sqrt(dx*dx + dy*dy), 1.0)
    _add_car(gs, "blue-0", BLUE_TEAM,
             [ax, ay, CAR_Z],
             [dx/norm*spd, dy/norm*spd, 0.], [dx/norm, dy/norm, 0.], boost=MAX_BOOST)
    _fill_dummies(gs, "blue-0")
    return gs


def build_slow_cant_reach() -> GameState:
    """Ball will score — agent is too far away to intervene.

    Ball starts at y∈[-4200,-4700] (420-920uu from goal), elevated ~150-300uu so
    air travel maintains speed, at vy∈[-800,-1200]. Arrival guaranteed in <1.2s.
    Agent is parked 3000-4500uu away in the orange half — physically impossible
    to reach in time. Agent receives full concede penalty.
    """
    gs = _base_gs()
    bx = random.uniform(-500., 500.)
    by = random.uniform(-4200., -4700.)
    bz = random.uniform(150., 300.)
    spd = random.uniform(800., 1200.)
    _set_ball(gs, [bx, by, bz], [random.uniform(-80., 80.), -spd, random.uniform(-50., 50.)])
    ay = by + random.uniform(3000., 4500.)         # far back in orange half
    _add_car(gs, "blue-0", BLUE_TEAM,
             [random.uniform(-1000., 1000.), ay, CAR_Z],
             [0., -100., 0.], [0., -1., 0.], boost=33.0)
    _fill_dummies(gs, "blue-0")
    return gs


def build_slow_can_reach() -> GameState:
    """Ball heading toward own goal at medium speed; agent must move laterally to intercept.

    Ball at y∈[-3000,-3800] elevated ~150uu so it maintains speed, at vy∈[-600,-900].
    Arrival ~1.5-4s.  Agent is offset 400-900uu laterally from ball's path and
    600-1200uu behind the ball — has to drive and angle, not just sit in the way.
    """
    gs = _base_gs()
    bx = random.uniform(-600., 600.)
    by = random.uniform(-3000., -3800.)
    bz = random.uniform(100., 200.)
    spd = random.uniform(600., 900.)
    _set_ball(gs, [bx, by, bz], [random.uniform(-60., 60.), -spd, random.uniform(-30., 30.)])
    # Offset laterally so agent must move across — not just stand in front
    side = random.choice([-1., 1.])
    ax = bx + side * random.uniform(400., 900.)
    ay = by + random.uniform(600., 1200.)          # orange side of ball
    car_spd = random.uniform(400., 700.)
    _add_car(gs, "blue-0", BLUE_TEAM,
             [ax, ay, CAR_Z],
             [0., -car_spd, 0.], [0., -1., 0.], boost=MAX_BOOST)
    _fill_dummies(gs, "blue-0")
    return gs


def build_bouncy_fast() -> GameState:
    """Fast elevated ball toward own goal; agent has a chance but must hustle.

    Ball at y∈[-2500,-3500] elevated 200-450uu, vy∈[-1200,-1800] — arrives in
    1.4-3s after a low bounce.  Agent is 1000-2500uu away offset laterally,
    heading toward ball's projected path. Newborn agents will miss most saves;
    good agents intercept by reading trajectory.
    """
    gs = _base_gs()
    bx = random.uniform(-700., 700.)
    by = random.uniform(-2500., -3500.)
    bz = random.uniform(200., 450.)
    spd_y = -random.uniform(1200., 1800.)
    spd_z = -random.uniform(100., 250.)
    spd_x = random.uniform(-200., 200.)
    _set_ball(gs, [bx, by, bz], [spd_x, spd_y, spd_z])
    # Agent offset — must read the ball and drive to intercept
    side = random.choice([-1., 1.])
    ax = bx + side * random.uniform(500., 1200.)
    ay = by + random.uniform(1000., 2500.)
    car_spd = random.uniform(700., 1100.)
    dx, dy = bx - ax, by - ay
    norm = max(np.sqrt(dx*dx + dy*dy), 1.0)
    _add_car(gs, "blue-0", BLUE_TEAM,
             [ax, ay, CAR_Z],
             [dx/norm*car_spd, dy/norm*car_spd, 0.], [dx/norm, dy/norm, 0.], boost=MAX_BOOST)
    _fill_dummies(gs, "blue-0")
    return gs


DRILL_BUILDERS = {
    "observe_goal":    build_observe_goal,
    "easy_tap":        build_easy_tap,
    "shot_window":     build_shot_window,
    "slow_cant_reach": build_slow_cant_reach,
    "slow_can_reach":  build_slow_can_reach,
    "bouncy_fast":     build_bouncy_fast,
}

# ---------------------------------------------------------------------------
# Per-profile env factory
# ---------------------------------------------------------------------------

def _make_profile_env(profile_name: str, global_profiles: Dict, profile_names: List[str]) -> EngineEnvAdapter:
    """Create a fresh env whose team spec contains only `profile_name`.

    All 6 car slots will be assigned this profile — the 5 dummies don't
    act, but their obs/reward computations use the correct profile one-hot.
    """
    single_spec = {_DRILL_TEAM: [profile_name]}

    engine = RocketSimEngine(rlbot_delay=True)
    action_parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)
    ll_obs_builder = AdvancedObs(profile_names=profile_names)

    reward_fn = TwoTeamAssignedHotswapRewardAdapter(
        global_profiles=global_profiles,
        team_specs=single_spec,
        blue_team_name=_DRILL_TEAM,
        orange_team_name=_DRILL_TEAM,
    )

    # Minimal AC adapters — no HL gradient updates during drills
    ac_cfg = ACConfig()
    ac_blue   = HotswapACAdapter(ACProfilePolicy([profile_name], cfg=ac_cfg))
    ac_orange = HotswapACAdapter(ACProfilePolicy([profile_name], cfg=ac_cfg))

    return EngineEnvAdapter(
        engine=engine,
        action_parser=action_parser,
        reward_function=reward_fn,
        ll_obs_builder=ll_obs_builder,
        blue_size=3,
        orange_size=3,
        blue_team_name=_DRILL_TEAM,
        orange_team_name=_DRILL_TEAM,
        team_specs=single_spec,
        global_profiles=global_profiles,
        ac_adapter_blue=ac_blue,
        ac_adapter_orange=ac_orange,
    )


# ---------------------------------------------------------------------------
# Drill runner
# ---------------------------------------------------------------------------

def _run_drill(drill: Dict, agent: PPOAgent, env: EngineEnvAdapter,
               pbar_outer: tqdm, profile_name: str,
               logger: Optional[LowLevelLogger], match_id_start: int) -> Dict[str, float]:
    name         = drill["name"]
    reps         = drill["reps"]
    observe_only = drill["observe_only"]
    defensive    = drill["defensive"]   # True = ball heading to agent's own goal
    builder      = DRILL_BUILDERS[name]
    train_aid    = "blue-0"

    total_reward = 0.0
    total_goals  = 0
    total_ticks  = 0
    obs_size     = agent.buffer.obs.shape[1]
    match_id     = match_id_start

    for _ in range(reps):
        gs = builder()
        _, info = env.reset(initial_state=gs)
        agent_ids = list(env.state.cars.keys())
        obs = info["ll_obs"].get(train_aid, np.zeros(obs_size, dtype=np.float32))

        ep_reward  = 0.0
        done       = False
        _t         = 0
        dep_ticks  = 0.0   # ticks spent by training agent this episode

        for _t in range(TICKS_PER_DRILL):
            actions     = {aid: np.array([NULL_ACTION_IDX], dtype=np.int64) for aid in agent_ids}
            actions_idx = {aid: NULL_ACTION_IDX for aid in agent_ids}

            if not observe_only:
                act, logp, val = agent.act(obs)
                actions[train_aid]     = np.array([act], dtype=np.int64)
                actions_idx[train_aid] = act

            _, _, done, step_info = env.step(actions, shared_info={"actions_idx": actions_idx})
            dep_ticks += 1.0

            if not observe_only:
                rew      = float(step_info.get("rewards", {}).get(train_aid, 0.0))
                next_obs = step_info.get("ll_obs", {}).get(train_aid, obs)
                agent.store(obs, act, logp, rew, val, done)
                ep_reward += rew
                obs = next_obs

            if done:
                break

        scored = done  # episode ends via goal_scored flag
        total_reward += ep_reward
        if scored:
            total_goals += 1
        total_ticks += _t + 1

        if logger is not None:
            # contrib: training agent's profile gets all the deployment time
            contrib = {profile_name: dep_ticks / max(dep_ticks, 1.0)}
            # Defensive drills: a goal means the opponent scored on the agent (concede).
            # Offensive drills: a goal means the agent scored.
            b_score = 0 if defensive else (1 if scored else 0)
            o_score = (1 if scored else 0) if defensive else 0
            logger.log_match(
                match_id=match_id,
                blue_team=f"tw_{name}",
                orange_team="dummy",
                blue_score=b_score,
                orange_score=o_score,
                blue_switches=0,
                orange_switches=0,
                contrib_blue=contrib,
                contrib_orange={},
            )
        match_id += 1

        pbar_outer.update(1)
        pbar_outer.set_postfix_str(f"{name} rew={ep_reward:.3f}")

    return {
        "reps":        reps,
        "mean_reward": total_reward / max(reps, 1),
        "goal_rate":   total_goals  / max(reps, 1),
        "mean_ticks":  total_ticks  / max(reps, 1),
        "match_id_end": match_id,
    }


# ---------------------------------------------------------------------------
# Public API: called from RL.py
# ---------------------------------------------------------------------------

def run_training_wheels(
    ppo_players: Dict[str, PPOAgent],
    global_profiles: Dict,
    profile_names: List[str],
    log_path: Optional[str] = None,
) -> None:
    """Run the full drill curriculum for every profile in `ppo_players`.

    Each profile gets its own env so that the profile one-hot in AdvancedObs
    is always correct.  After all drills, `ppo_players` weights are updated
    in-place — the caller (RL.py) handles saving.

    If `log_path` is provided a LowLevelLogger CSV is written there with one
    row per drill episode so analysis_combined.ipynb can inspect drill results.
    Blue team is set to "tw_<drill_name>" and blue_score=1 when a goal was scored.
    """
    logger: Optional[LowLevelLogger] = None
    if log_path is not None:
        logger = LowLevelLogger(path=log_path, profile_names=profile_names)

    total_reps = sum(d["reps"] for d in DRILLS) * len(profile_names)
    pbar = tqdm(total=total_reps, desc="Training wheels", dynamic_ncols=True)

    all_results: Dict[str, Dict] = {}
    match_id = 0

    for profile_name in profile_names:
        agent = ppo_players[profile_name]
        env   = _make_profile_env(profile_name, global_profiles, profile_names)

        profile_results = {}
        for drill in DRILLS:
            stats = _run_drill(drill, agent, env, pbar, profile_name, logger, match_id)
            profile_results[drill["name"]] = stats
            match_id = stats["match_id_end"]

        all_results[profile_name] = profile_results
        del env   # release RocketSimEngine

    pbar.close()
    if logger is not None:
        logger.close()

    print("\n[training_wheels] Summary")
    for pname, drills in all_results.items():
        mean_rew = np.mean([d["mean_reward"] for d in drills.values()])
        goal_rt  = np.mean([d["goal_rate"]   for d in drills.values()])
        print(f"  {pname:4s}  mean_rew={mean_rew:+.4f}  goal_rate={goal_rt:.3f}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training wheels — standalone run")
    parser.add_argument("--agents", default="agents")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--device", default="gpu" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    agent_dir = Path(args.agents)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device    = args.device

    global_profiles, team_specs = build_globals()
    profile_names = list(global_profiles.keys())

    # Probe obs size using a throwaway env
    print("Probing obs size...")
    _probe_env = _make_profile_env(profile_names[0], global_profiles, profile_names)
    _obs, _ = _probe_env.reset(initial_state=build_easy_tap())
    obs_size = _obs.shape[0]
    del _probe_env
    print(f"  obs_size={obs_size}")

    # Load or init agents
    ppo_players: Dict[str, PPOAgent] = {}
    for pname in profile_names:
        ag = PPOAgent(obs_size=obs_size, n_actions=90, device=device)
        ckpt = agent_dir / f"{pname}.pt"
        if ckpt.exists():
            ag.net.load_state_dict(torch.load(str(ckpt), map_location=device))
        ppo_players[pname] = ag

    tw_log = str(out_dir / "training_wheels_log.csv")
    run_training_wheels(ppo_players, global_profiles, profile_names, log_path=tw_log)
    print(f"Log written to {tw_log}")

    # Save
    agent_dir.mkdir(parents=True, exist_ok=True)
    for pname, ag in ppo_players.items():
        torch.save(ag.net.state_dict(), str(agent_dir / f"{pname}.pt"))
        torch.save(ag.opt.state_dict(),  str(agent_dir / f"{pname}.opt.pt"))
    print(f"Saved to {agent_dir}/")


if __name__ == "__main__":
    main()
