"""Shared curriculum schedules and causal intervention sampling."""

POSITION_NAMES = ('early', 'middle', 'late')
STREAK_TARGETS = (0, 3, 8)

def choose_intervention(draw, n_events, depth, maintenance):
    """Spread teacher windows over early/middle/late and long KEEP streaks."""
    n_decisions = max(0, int(n_events) - 1) + int(maintenance)
    if n_decisions < 1:
        raise ValueError('no_decision_steps')
    depth = min(int(depth), n_decisions)
    n_starts = n_decisions - depth + 1
    pos_i = int(draw) % 3
    streak_i = (int(draw) // 3) % 3
    target_streak = STREAK_TARGETS[streak_i]
    thirds = [[] for _ in range(3)]
    for index in range(n_starts):
        if n_starts == 1:
            thirds[0].append(index)
        else:
            thirds[0 if index < n_starts / 3 else (1 if index < 2 * n_starts / 3 else 2)].append(index)
    choices = thirds[pos_i] or [index for group in thirds for index in group]
    eligible = [index for index in choices if index >= target_streak] or choices
    start_index = eligible[(int(draw) * 2654435761) % len(eligible)]
    start = start_index + 1
    return {
        'start': start,
        'depth': depth,
        'position_bucket': POSITION_NAMES[pos_i],
        'target_keep_streak': target_streak,
        'n_decisions': n_decisions,
        'keep_prefix_from': max(1, start - target_streak),
    }

def least_harm_shrink(records, max_harm=None):
    """Pick a realized SHRINK that saves cost and hurts the least vs KEEP."""
    best=None
    for rec in records or []:
        realized=list(rec.get('realized_sequence') or rec.get('sequence') or [])
        if 1 not in realized: continue
        savings=rec.get('cost_savings_vs_keep')
        if savings is None or float(savings)<=0: continue
        harm=float(rec.get('max_question_harm') or 0.)
        if max_harm is not None and harm>float(max_harm): continue
        first=next(index for index,action in enumerate(realized) if action==1)
        key=(harm,-float(savings),tuple(rec.get('sequence') or realized))
        if best is None or key<best[0]:
            best=(key,rec,realized[first],first,harm)
    if best is None: return None
    return {'record':best[1],'action':best[2],'divergence':best[3],'harm':best[4],
            'sequence':list(best[1].get('sequence') or best[1].get('realized_sequence'))}

def arm_flags(arm):
    """Curriculum opens Writer, uses slot reweight, and schedules forced unkeep."""
    curriculum = arm == 'curriculum'
    return {
        'quota': arm in ('quota','quota_codec','quota_slot'),
        'codec': curriculum or arm in ('codec','slot','quota_codec','quota_slot','dynpen_slot'),
        'slot': curriculum or arm in ('slot','quota_slot','dynpen_slot'),
        'dynpen': arm in ('dynpen','dynpen_slot'),
        'curriculum': curriculum,
    }

def force_unkeep_rate(progress, start=0.75, end=0.15):
    progress=min(1.,max(0.,float(progress)))
    return start+(end-start)*progress

def unkeep_target_rate(progress, start=0.50, end=0.22):
    progress=min(1.,max(0.,float(progress)))
    return start+(end-start)*progress

def rate_band_penalty(ema_unkeep, target, band=0.12):
    """Return (side, scale). side is keep/unkeep/none; scale in [0, 1]."""
    low=float(target)-float(band); high=float(target)+float(band)
    value=float(ema_unkeep)
    if value<low:
        return 'keep', min(1., (low-value)/max(float(band),1e-6))
    if value>high:
        return 'unkeep', min(1., (value-high)/max(float(band),1e-6))
    return 'none', 0.

def dynamic_keep_penalty(keep_rate_ema, keep_streak_before, target=0.5):
    """Tax P(KEEP) when the recent policy is more KEEP-heavy than the target."""
    excess=max(0.,float(keep_rate_ema)-float(target))/max(1e-6,1.-float(target))
    streak=min((int(keep_streak_before)+1)/8.,1.)
    return 0.5*excess+0.3*streak
