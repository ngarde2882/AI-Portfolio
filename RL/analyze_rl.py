# -*- coding: utf-8 -*-
"""
Rocket League Hierarchical RL Training Log Analysis
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

BASE = r"C:/Users/nick2/Desktop/School Stuff/stats/AI-Portfolio/RL/out"

# ============================================================
# HELPERS
# ============================================================
def parse_tuple(s):
    """Parse '(blue_val, orange_val)' → (float, float)"""
    try:
        m = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s))
        if len(m) >= 2:
            return float(m[0]), float(m[1])
    except:
        pass
    return (0.0, 0.0)

def section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data files...")
ll  = pd.read_csv(f"{BASE}/low_level_log.csv")
rc  = pd.read_csv(f"{BASE}/reward_contrib.csv")

# tournament_logs has no header — use the same column names as low_level_log
trn_raw = pd.read_csv(f"{BASE}/tournament_logs.csv", header=None, skip_blank_lines=True)
trn_raw = trn_raw.dropna(how='all')
# assign columns from ll
trn_raw.columns = ll.columns[:len(trn_raw.columns)]
trn = trn_raw.copy()
trn['match_id'] = pd.to_numeric(trn['match_id'], errors='coerce')
trn = trn.dropna(subset=['match_id'])
trn['match_id'] = trn['match_id'].astype(int)
for col in ['blue_score','orange_score','goal_diff','blue_switches','orange_switches']:
    if col in trn.columns:
        trn[col] = pd.to_numeric(trn[col], errors='coerce').fillna(0).astype(int)

print(f"  low_level_log   : {len(ll)} rows")
print(f"  reward_contrib  : {len(rc)} rows")
print(f"  tournament_logs : {len(trn)} rows")

# ============================================================
# SECTION 1 – LOW-LEVEL GAUNTLET STATS
# ============================================================
section("1. LOW-LEVEL GAUNTLET STATS")

total_games = len(ll)
scoring     = (ll['goal_diff'] != 0).sum()
null_games  = ((ll['blue_score'] == 0) & (ll['orange_score'] == 0) & (ll['goal_diff'] == 0)).sum()
active_ns   = ((ll['blue_score'] == 0) & (ll['orange_score'] == 0) & (ll['goal_diff'] != 0)).sum()

print(f"\n--- Basic Counts ---")
print(f"  Total games         : {total_games}")
print(f"  Scoring games       : {scoring}  ({100*scoring/total_games:.1f}%)")
print(f"  True-inactive (0-0) : {null_games}  ({100*null_games/total_games:.1f}%)")
print(f"  Active non-scoring  : {active_ns}  (goal_diff!=0 but 0-0 scores; should be 0 in theory)")

# goal_diff histogram
print(f"\n--- goal_diff Distribution ---")
gd_counts = ll['goal_diff'].value_counts().sort_index()
for gd, cnt in gd_counts.items():
    bar = '#' * int(cnt / total_games * 200)
    print(f"  {gd:+3d}  {cnt:5d} ({100*cnt/total_games:5.1f}%)  {bar}")

# Win rate by team
print(f"\n--- Win Rate by blue_team ---")
for team, grp in ll.groupby('blue_team'):
    wins  = (grp['winner'] == 'BLUE').sum()
    ties  = (grp['winner'] == 'TIE').sum()
    losses= (grp['winner'] == 'ORANGE').sum()
    n     = len(grp)
    print(f"  {team:<30s}  W:{wins:4d}({100*wins/n:5.1f}%)  T:{ties:4d}({100*ties/n:5.1f}%)  L:{losses:4d}({100*losses/n:5.1f}%)  n={n}")

# Profile contribution stats
PROFILES = ['s0','s1','s2','s3','s4','s5','p1','p2','p3','d1','d2','d3','d4']
prof_cols = {p: f'contrib_{p}' for p in PROFILES}

print(f"\n--- Profile Contribution (avg blue+orange frac, non-zero rate) ---")
prof_stats = {}
for p, col in prof_cols.items():
    if col not in ll.columns:
        continue
    parsed = ll[col].apply(parse_tuple)
    blue_frac   = parsed.apply(lambda x: x[0])
    orange_frac = parsed.apply(lambda x: x[1])
    nonzero = ((blue_frac != 0) | (orange_frac != 0)).sum()
    avg_b = blue_frac.mean()
    avg_o = orange_frac.mean()
    prof_stats[p] = {'avg_blue': avg_b, 'avg_orange': avg_o, 'nonzero': nonzero}
    print(f"  {p:<5s}  avg_blue={avg_b:.4f}  avg_orange={avg_o:.4f}  "
          f"nonzero_games={nonzero:5d} ({100*nonzero/total_games:5.1f}%)")

# HL switch rates
print(f"\n--- HL Switch Rates ---")
print(f"  blue_switches  mean={ll['blue_switches'].mean():.3f}  "
      f"std={ll['blue_switches'].std():.3f}  max={ll['blue_switches'].max()}")
print(f"  orange_switches mean={ll['orange_switches'].mean():.3f}  "
      f"std={ll['orange_switches'].std():.3f}  max={ll['orange_switches'].max()}")

sw_b = ll['blue_switches'].value_counts().sort_index()
sw_o = ll['orange_switches'].value_counts().sort_index()
print(f"  blue_switches  distribution: { {k:v for k,v in sw_b.items()} }")
print(f"  orange_switches distribution: { {k:v for k,v in sw_o.items()} }")

# Trend over match_id
print(f"\n--- Trend over match_id (early/mid/late thirds) ---")
n3 = total_games // 3
thirds = [
    ('early', ll.iloc[:n3]),
    ('mid',   ll.iloc[n3:2*n3]),
    ('late',  ll.iloc[2*n3:]),
]
for label, grp in thirds:
    sr = (grp['goal_diff'] != 0).mean()
    agd = grp['goal_diff'].abs().mean()
    null_r = ((grp['blue_score']==0) & (grp['orange_score']==0)).mean()
    mid_id = grp['match_id'].median()
    print(f"  {label:<6s}  mid_match_id={mid_id:6.0f}  "
          f"scoring_rate={sr:.3f}  avg_|goal_diff|={agd:.3f}  null_rate={null_r:.3f}")

# ============================================================
# SECTION 2 – REWARD CONTRIBUTIONS
# ============================================================
section("2. REWARD CONTRIBUTIONS")

# Identify reward tuple columns
reward_tuple_cols = [c for c in rc.columns if c.startswith('r_') or c == 'postgame_contrib_goal_diff']
scalar_reward_cols = ['ac_tick_reward_blue', 'ac_tick_reward_orange']

print(f"\n--- Parsing reward tuples ({len(reward_tuple_cols)} columns) ---")
reward_blue  = {}
reward_orange = {}
for col in reward_tuple_cols:
    parsed = rc[col].apply(parse_tuple)
    reward_blue[col]   = parsed.apply(lambda x: x[0])
    reward_orange[col] = parsed.apply(lambda x: x[1])

rb_df = pd.DataFrame(reward_blue)
ro_df = pd.DataFrame(reward_orange)

# Per-game per-team averages
print(f"\n--- Reward Type Magnitudes (mean per game) ---")
rb_means = rb_df.mean().sort_values(key=abs, ascending=False)
ro_means = ro_df.mean().sort_values(key=abs, ascending=False)

print(f"  {'Reward':<35s}  {'BLUE mean':>12s}  {'ORANGE mean':>12s}")
all_cols = reward_tuple_cols
for col in sorted(all_cols, key=lambda c: abs(rb_df[c].mean()) + abs(ro_df[c].mean()), reverse=True):
    bm = rb_df[col].mean()
    om = ro_df[col].mean()
    print(f"  {col:<35s}  {bm:>12.4f}  {om:>12.4f}")

# Near-zero / dead signals
ZERO_THRESH = 0.001
dead = [c for c in all_cols if abs(rb_df[c].mean()) < ZERO_THRESH and abs(ro_df[c].mean()) < ZERO_THRESH]
print(f"\n--- Near-zero (dead) signals (|mean|<{ZERO_THRESH}) ---")
for d in dead:
    print(f"  {d}")

# ac_tick_reward
print(f"\n--- ac_tick_reward_blue/orange ---")
for col in scalar_reward_cols:
    s = rc[col].dropna()
    pct_pos = (s > 0).mean() * 100
    print(f"  {col:<30s}  mean={s.mean():.4f}  std={s.std():.4f}  "
          f"min={s.min():.4f}  max={s.max():.4f}  pct_positive={pct_pos:.1f}%")

# Correlation with goal_diff
print(f"\n--- Correlation: reward type vs goal_diff ---")
corr_data = rb_df.copy()
corr_data['goal_diff'] = rc['goal_diff'].values
corrs_b = corr_data.corr()['goal_diff'].drop('goal_diff').sort_values(key=abs, ascending=False)

corr_data_o = ro_df.copy()
corr_data_o['goal_diff'] = rc['goal_diff'].values
corrs_o = corr_data_o.corr()['goal_diff'].drop('goal_diff').sort_values(key=abs, ascending=False)

print(f"  {'Reward':<35s}  {'BLUE corr':>10s}  {'ORANGE corr':>12s}")
for col in corrs_b.index:
    print(f"  {col:<35s}  {corrs_b[col]:>10.4f}  {corrs_o.get(col,0):>12.4f}")

# reward_scale trend
print(f"\n--- reward_scale trend over match_id ---")
if 'reward_scale' in rc.columns:
    rs_by_mid = rc.groupby('match_id')['reward_scale'].first()
    min_id, max_id = rs_by_mid.index.min(), rs_by_mid.index.max()
    buckets = np.array_split(rs_by_mid.values, 5)
    ids     = np.array_split(rs_by_mid.index.values, 5)
    for i, (bid, bv) in enumerate(zip(ids, buckets)):
        print(f"  segment {i+1}  match_id ~{bid[0]}-{bid[-1]:4d}  "
              f"reward_scale mean={np.mean(bv):.2f}  min={np.min(bv):.2f}  max={np.max(bv):.2f}")

# Scoring vs null reward comparison
print(f"\n--- Reward profile: Scoring vs Null (0-0) games ---")
scoring_mask = rc['goal_diff'] != 0
null_mask    = (rc['blue_score'] == 0) & (rc['orange_score'] == 0) & (rc['goal_diff'] == 0)

print(f"  {'Reward':<35s}  {'BLUE(score)':>11s}  {'BLUE(null)':>10s}  "
      f"{'OR(score)':>10s}  {'OR(null)':>10s}")
for col in all_cols:
    bs = rb_df[col][scoring_mask].mean()
    bn = rb_df[col][null_mask].mean()
    os = ro_df[col][scoring_mask].mean()
    on = ro_df[col][null_mask].mean()
    if abs(bs) + abs(bn) + abs(os) + abs(on) > 0.005:
        print(f"  {col:<35s}  {bs:>11.4f}  {bn:>10.4f}  {os:>10.4f}  {on:>10.4f}")

# ============================================================
# SECTION 3 – TOURNAMENT ANALYSIS
# ============================================================
section("3. TOURNAMENT ANALYSIS")

print(f"\n--- Tournament Overview ---")
print(f"  Total games: {len(trn)}")
print(f"  Unique matchups: {trn.groupby(['blue_team','orange_team']).ngroups}")

# Team win rates — both as blue and orange
teams = set(trn['blue_team'].tolist() + trn['orange_team'].tolist())
team_stats = {}
for team in sorted(teams):
    as_blue   = trn[trn['blue_team']   == team]
    as_orange = trn[trn['orange_team'] == team]
    w = (as_blue['winner']=='BLUE').sum() + (as_orange['winner']=='ORANGE').sum()
    t = (as_blue['winner']=='TIE').sum()  + (as_orange['winner']=='TIE').sum()
    l = (as_blue['winner']=='ORANGE').sum() + (as_orange['winner']=='BLUE').sum()
    n = len(as_blue) + len(as_orange)
    gf= as_blue['blue_score'].sum() + as_orange['orange_score'].sum()
    ga= as_blue['orange_score'].sum() + as_orange['blue_score'].sum()
    team_stats[team] = {'W':w,'T':t,'L':l,'n':n,'GF':gf,'GA':ga}

print(f"\n  {'Team':<30s}  W    T    L    n   GF  GA  WR%   NullR%")
for team, s in sorted(team_stats.items(), key=lambda x: -x[1]['W']):
    n = s['n']
    wr = 100*s['W']/n if n else 0
    null_r = 100*(s['T']) / n if n else 0  # ties ≠ null but indicative
    print(f"  {team:<30s}  {s['W']:<4d} {s['T']:<4d} {s['L']:<4d} {n:<4d} "
          f"{s['GF']:<3d} {s['GA']:<3d} {wr:5.1f}%  {null_r:5.1f}%")

# Scores & ties
print(f"\n--- Score Totals ---")
total_goals = trn['blue_score'].sum() + trn['orange_score'].sum()
ties = (trn['winner'] == 'TIE').sum()
print(f"  Total goals scored : {total_goals}")
print(f"  Ties               : {ties} ({100*ties/len(trn):.1f}%)")
print(f"  Avg goals per game : {total_goals/len(trn):.3f}")

# Switch activity
print(f"\n--- Switch Activity (gauntlet vs tournament) ---")
g_sw_b = ll['blue_switches'].mean()
g_sw_o = ll['orange_switches'].mean()
t_sw_b = trn['blue_switches'].mean()
t_sw_o = trn['orange_switches'].mean()
print(f"  Gauntlet  : blue={g_sw_b:.3f}  orange={g_sw_o:.3f}")
print(f"  Tournament: blue={t_sw_b:.3f}  orange={t_sw_o:.3f}")

# ============================================================
# SECTION 4 – HIGH-LEVEL LOG SPOT ANALYSIS
# ============================================================
section("4. HIGH-LEVEL LOG SPOT ANALYSIS")

hl_files = [
    (f"{BASE}/tournament_high_logs/1606)team3_balancevteam2_striker_heavy.csv",
     "Match 1606: team3_balance vs team2_striker_heavy"),
    (f"{BASE}/tournament_high_logs/1621)team1_striker_balancevteam5_balance2.csv",
     "Match 1621: team1_striker_balance vs team5_balance2"),
]

for fpath, label in hl_files:
    print(f"\n{'-'*65}")
    print(f"  {label}")
    print(f"{'-'*65}")
    hl = pd.read_csv(fpath)
    print(f"  Rows: {len(hl)}  Ticks: {hl['tick'].max()}  Duration: {hl['time_s'].max():.1f}s")

    # Ball py — field control (negative = blue side, positive = orange side)
    bp = hl['ball_py']
    print(f"\n  Ball py (field control) — range: [{bp.min():.0f}, {bp.max():.0f}]")
    segments = np.array_split(hl.index, 5)
    for i, idx in enumerate(segments):
        t0 = hl.loc[idx[0], 'time_s']
        t1 = hl.loc[idx[-1], 'time_s']
        avg_py = bp.loc[idx].mean()
        side = "BLUE side" if avg_py < -500 else "ORANGE side" if avg_py > 500 else "mid-field"
        print(f"    seg {i+1} ({t0:.1f}s-{t1:.1f}s)  avg_ball_py={avg_py:8.1f}  ({side})")

    # Ball speed
    bspd = np.sqrt(hl['ball_vx']**2 + hl['ball_vy']**2 + hl['ball_vz']**2)
    print(f"\n  Ball speed — mean={bspd.mean():.1f}  max={bspd.max():.1f}  "
          f"pct_stationary(<50)={(bspd<50).mean()*100:.1f}%")

    # Inactivity periods (ball_py near 0 and ball slow)
    inactive = ((bspd < 100) & (bp.abs() < 500))
    print(f"  Ball near-center+slow ticks: {inactive.sum()} ({100*inactive.mean():.1f}%)")

    # Profile switches per team
    print(f"\n  Profile Switches per player:")
    for side in ['blue', 'orange']:
        for pid in ['0','1','2']:
            col = f"{side}-{pid}_profile"
            sw_col = f"{side}-{pid}_switches_window"
            if col in hl.columns:
                prof_counts = hl[col].value_counts().to_dict()
                total_sw = hl[sw_col].max() if sw_col in hl.columns else 'N/A'
                # count actual switches as profile change events
                sw_events = (hl[col] != hl[col].shift()).sum() - 1
                print(f"    {side}-{pid}:  profiles used={list(prof_counts.keys())}  "
                      f"transitions={sw_events}  max_switch_ctr={total_sw}")

    # Car speeds
    print(f"\n  Car speeds (mean over game):")
    for side in ['blue', 'orange']:
        for pid in ['0','1','2']:
            vx, vy, vz = f"{side}-{pid}_vx", f"{side}-{pid}_vy", f"{side}-{pid}_vz"
            if vx in hl.columns:
                spd = np.sqrt(hl[vx]**2 + hl[vy]**2 + hl[vz]**2)
                print(f"    {side}-{pid}:  mean={spd.mean():.1f}  max={spd.max():.1f}")

    # Final score
    fs_b = hl['blue_score'].iloc[-1]
    fs_o = hl['orange_score'].iloc[-1]
    print(f"\n  Final score: BLUE {fs_b} - {fs_o} ORANGE")

# ============================================================
# SECTION 5 - CROSS-FILE INSIGHTS
# ============================================================
section("5. CROSS-FILE INSIGHTS")

print(f"\n--- 5a. Gauntlet scoring rate by blue_team -> tournament performance ---")

# Gauntlet scoring rate per blue team
gaunt_sr = ll.groupby('blue_team').apply(
    lambda g: pd.Series({
        'scoring_rate': (g['goal_diff'] != 0).mean(),
        'null_rate':    ((g['blue_score']==0)&(g['orange_score']==0)).mean(),
        'avg_goal_diff': g['goal_diff'].mean(),
        'n_games': len(g)
    })
).reset_index()

gaunt_sr = gaunt_sr.sort_values('scoring_rate', ascending=False)
print(f"\n  {'Team':<30s}  scoring_rt  null_rt  avg_gdiff  n_games")
for _, row in gaunt_sr.iterrows():
    print(f"  {row['blue_team']:<30s}  {row['scoring_rate']:.3f}      "
          f"{row['null_rate']:.3f}    {row['avg_goal_diff']:+.3f}     {int(row['n_games'])}")

# Compare with tournament wins
print(f"\n  Tournament wins for top gauntlet scorers:")
for _, row in gaunt_sr.head(4).iterrows():
    tm = row['blue_team']
    if tm in team_stats:
        ts = team_stats[tm]
        print(f"    {tm:<30s}  gauntlet_sr={row['scoring_rate']:.3f}  "
              f"tournament W-T-L: {ts['W']}-{ts['T']}-{ts['L']}")
    else:
        print(f"    {tm:<30s}  not in tournament")

print(f"\n--- 5b. Teams with consistently low null rates ---")
low_null = gaunt_sr[gaunt_sr['null_rate'] < 0.6].sort_values('null_rate')
for _, row in low_null.iterrows():
    print(f"  {row['blue_team']:<30s}  null_rate={row['null_rate']:.3f}  scoring_rate={row['scoring_rate']:.3f}")

print(f"\n--- 5c. s5 profile analysis (dist_to_ball improvement) ---")
s5_col = 'contrib_s5'
if s5_col in ll.columns:
    parsed_s5 = ll[s5_col].apply(parse_tuple)
    s5_blue   = parsed_s5.apply(lambda x: x[0])
    s5_orange = parsed_s5.apply(lambda x: x[1])
    s5_nonzero = ((s5_blue != 0) | (s5_orange != 0))

    # Split into early/late by match_id
    mid_point = ll['match_id'].median()
    early_mask = ll['match_id'] <= mid_point
    late_mask  = ll['match_id'] > mid_point

    nz_early = s5_nonzero[early_mask].mean()
    nz_late  = s5_nonzero[late_mask].mean()
    avg_b_early = s5_blue[early_mask].mean()
    avg_b_late  = s5_blue[late_mask].mean()

    print(f"  s5 non-zero contrib rate  early half={nz_early:.4f}  late half={nz_late:.4f}  "
          f"delta={nz_late-nz_early:+.4f}")
    print(f"  s5 avg blue_frac          early half={avg_b_early:.4f}  late half={avg_b_late:.4f}")

    # Per-team analysis — is s5 more active on teams that use it?
    print(f"\n  s5 non-zero rate per blue_team (as blue):")
    for team, grp_idx in ll.groupby('blue_team').groups.items():
        grp = ll.loc[grp_idx]
        p5_b = s5_blue.loc[grp_idx]
        p5_o = s5_orange.loc[grp_idx]
        nz = ((p5_b != 0) | (p5_o != 0)).mean()
        if nz > 0:
            print(f"    {team:<30s}  s5_nonzero_rate={nz:.4f}")
else:
    print("  contrib_s5 column not found")

# Extra: which reward signals correlate most with scoring?
print(f"\n--- 5d. Top reward signals predictive of scoring (|corr| > 0.1) ---")
for col in corrs_b.index:
    cb = corrs_b[col]
    co = corrs_o.get(col, 0)
    if abs(cb) > 0.1 or abs(co) > 0.1:
        print(f"  {col:<35s}  blue_corr={cb:+.4f}  orange_corr={co:+.4f}")

print(f"\n{'='*70}")
print("  ANALYSIS COMPLETE")
print(f"{'='*70}\n")
