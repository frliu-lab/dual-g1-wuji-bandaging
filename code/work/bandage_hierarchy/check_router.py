"""Small behavior checks for handover, continuity and stale-policy fallback."""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from controller_router import HandExpertRouter, HandProposal, Mode, Observation

r = HandExpertRouter(80)
zeros = np.zeros(80)
ones = np.ones(80)


def observation(t, **kwargs):
    values = dict(time_s=t, phase='wrap', body_valid=True, feet_supported=True,
                  arm_pose_stable=True, hand_contacts=(True, True, True, True),
                  hand_feedback_fresh=True, dressing_grasp_valid=True,
                  forearm_support_valid=True)
    values.update(kwargs)
    return Observation(**values)


assert r.step(observation(0., hand_contacts=(True, False, False, False)), zeros, None).mode == Mode.BODY
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
d2 = r2.step(observation(.35, hand_contacts=(True, True, False, True)), zeros, HandProposal(.35, ones, .95))
assert d2.mode == Mode.HOLD and np.array_equal(d2.finger_targets_rad, held)
r3 = HandExpertRouter(80)
for t in (.0, .05, .10, .15, .20, .25, .30):
    d3 = r3.step(observation(t, phase='place_dressing',
                             hand_contacts=(True, True, False, False)),
                 zeros, HandProposal(t, ones, .95))
assert d3.mode == Mode.HAND
r4 = HandExpertRouter(80)
for t in (.0, .05, .10, .15, .20, .25, .30):
    d4 = r4.step(observation(t, phase='place_dressing', dressing_grasp_valid=False),
                 zeros, HandProposal(t, ones, .95))
assert d4.mode == Mode.BODY
r5 = HandExpertRouter(80)
for t in (.0, .05, .10, .15, .20, .25, .30):
    d5 = r5.step(observation(t, forearm_support_valid=False), zeros,
                 HandProposal(t, ones, .95))
assert d5.mode == Mode.BODY
r6 = HandExpertRouter(80)
for t in (.0, .05, .10, .15, .20, .25, .30):
    d6 = r6.step(observation(t), zeros, HandProposal(t, ones, .95))
assert d6.mode == Mode.HAND
d6 = r6.step(observation(.35, forearm_support_valid=False), zeros,
             HandProposal(.35, ones, .95))
assert d6.mode == Mode.HOLD and d6.specialist_weight == 0.
trace_path=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5.json'
trace=json.loads(trace_path.read_text(encoding='utf8'))['conditions']['palm_candidate_path']['trace']
r7=HandExpertRouter(80)
for frame in trace:
    t=frame['time_s']
    decision=r7.step(observation(t, forearm_support_valid=frame['online_support_valid']),
                     zeros, HandProposal(t, ones, .95))
    assert decision.mode == Mode.BODY
assert not any(frame['online_support_valid'] for frame in trace)
passing_path=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0/dynamic_palm_support_wbc_comparison_follow0p5_wbcforce12p0_wbcfinger2p0.json'
passing_trace=json.loads(passing_path.read_text(encoding='utf8'))['conditions']['palm_candidate_path']['trace']
r8=HandExpertRouter(80)
first_valid=next(frame['time_s'] for frame in passing_trace if frame['online_support_valid'])
first_hand=None
for frame in passing_trace:
    t=frame['time_s']
    decision=r8.step(observation(t, forearm_support_valid=frame['online_support_valid']),
                     zeros, HandProposal(t, ones, .95))
    if decision.mode == Mode.HAND and first_hand is None:
        first_hand=t
    if t < first_valid+.25-1e-9:
        assert decision.mode == Mode.BODY
assert first_hand is not None and first_hand >= first_valid+.25-1e-9
# The positive replay isolates the support gate; the other three contacts,
# dressing grasp and hand proposal are synthetic fixtures, not validated skills.
r9=HandExpertRouter(80)
for frame in passing_trace:
    t=frame['time_s']
    decision=r9.step(observation(t, forearm_support_valid=frame['online_support_valid'],
                                 dressing_grasp_valid=False), zeros, HandProposal(t, ones, .95))
    assert decision.mode == Mode.BODY
print(f'router checks passed: failed/conditional support replays, first conditional hand switch {first_hand:.2f}s, stale fallback and loss guards')
