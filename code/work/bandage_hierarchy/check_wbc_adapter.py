"""Check four-palm WBC interface and explicit flex-contact refusal."""
from pathlib import Path
import sys
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC
from guided_flex_pilot import make_guided_xml

model = mujoco.MjModel.from_xml_path(str(ROOT / 'outputs/wuji_multiagent/bandage_v0/scene.xml'))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
wbc = JointWBC(model, data.qpos.copy(), palm_names=palms)
assert wbc.task_dim == 24 and len(wbc.palm_ids) == 4
assert JointWBC(model, data.qpos.copy()).task_dim == 12

flex_model = mujoco.MjModel.from_xml_string(make_guided_xml())
flex_data = mujoco.MjData(flex_model)
for _ in range(100):
    mujoco.mj_step(flex_model, flex_data)
assert any(c.geom1 < 0 or c.geom2 < 0 for c in flex_data.contact)
guard = object.__new__(JointWBC)
guard.m = flex_model
try:
    guard.contacts(flex_data)
except RuntimeError as exc:
    assert 'flex contacts' in str(exc)
else:
    raise AssertionError('JointWBC must refuse unsupported flex contacts')
print('WBC adapter checks passed: four palm tasks, legacy default, flex contact fail-closed')
