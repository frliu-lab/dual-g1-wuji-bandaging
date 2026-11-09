"""Check that the centerline responds to a movable spool.

This tests geometry only. A spatial tendon is not a cloth bandage.
"""
from pathlib import Path
import json
import mujoco

root = Path(__file__).resolve().parents[2]
scene = root / 'outputs/wuji_multiagent/bandage_v0/scene.xml'
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
tid = model.tendon('bandage_route_centerline').id
qid = model.joint('bandage_spool_free').qposadr[0]
baseline = float(data.ten_length[tid])
data.qpos[qid] += .1
mujoco.mj_forward(model, data)
shifted = float(data.ten_length[tid])
result = {
    'scope': 'Geometric centerline response to a 0.1 m spool displacement, not cloth or force validation',
    'baseline_length_m': baseline,
    'shifted_length_m': shifted,
    'length_change_m': shifted-baseline,
    'responds_to_spool_motion': abs(shifted-baseline) > .01,
}
(scene.parent / 'route_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
if not result['responds_to_spool_motion']:
    raise RuntimeError('Bandage centerline did not respond to spool motion')
