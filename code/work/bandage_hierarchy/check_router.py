"""Small behavior checks for handover, continuity and stale-policy fallback."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from controller_router import HandExpertRouter, HandProposal, Mode, Observation

r = HandExpertRouter(80)
zeros = np.zeros(80)
ones = np.ones(80)


def observation(t, **kwargs):
    values = dict(time_s=t, phase='wrap', body_valid=True, feet_supported=True,
                  arm_pose_stable=True, contact_confirmed=True, hand_feedback_fresh=True)
    values.update(kwargs)
    return Observation(**values)


assert r.step(observation(0., contact_confirmed=False), zeros, None).mode == Mode.BODY
for t in (.05, .10, .15, .20, .25, .30):
    decision = r.step(observation(t), zeros, HandProposal(t, ones, .95))
assert decision.mode == Mode.HAND and 0. < decision.specialist_weight < 1.
assert np.max(decision.finger_targets_rad) <= .8*.05+1e-9
last = decision.finger_targets_rad.copy()
decision = r.step(observation(.35), zeros, HandProposal(.20, ones, .95))
assert decision.mode == Mode.HOLD and np.array_equal(decision.finger_targets_rad, last)
decision = r.step(observation(.40, feet_supported=False), zeros, None)
assert decision.mode == Mode.HOLD and np.array_equal(decision.finger_targets_rad, last)
r2 = HandExpertRouter(80)
for t in (.0, .05, .10, .15, .20, .25, .30):
    d2 = r2.step(observation(t), zeros, HandProposal(t, ones, .95))
assert d2.mode == Mode.HAND
held = d2.finger_targets_rad.copy()
d2 = r2.step(observation(.35, contact_confirmed=False), zeros, HandProposal(.35, ones, .95))
assert d2.mode == Mode.HOLD and np.array_equal(d2.finger_targets_rad, held)
print('router checks passed: contact gate, handover rate limit, stale fallback, support and contact-loss guards')
