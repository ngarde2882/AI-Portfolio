source activate rl_env

python - <<'EOF'
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.state_mutators import KickoffMutator
print("Instantiating engine...")
engine = RocketSimEngine(rlbot_delay=True)
print("Engine OK")
EOF