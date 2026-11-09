"""One cold four-palm WBC solve in the bandaging scene, without flex cloth."""
import json
from pathlib import Path
import sys
import time
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC

model = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for _ in range(round(.1/model.opt.timestep)):
    mujoco.mj_step(model, data)
palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
wbc = JointWBC(model, data.qpos.copy(), palm_names=palms)
started = time.perf_counter()
try:
    ctrl, meta, _ = wbc.solve(data)
    legacy_ctrl, legacy_meta, _ = JointWBC(model, data.qpos.copy()).solve(data)
    report = {'scope': 'Cold QP solve only; no rollout, cloth or hand specialist',
              'four_palm_tasks': wbc.task_dim == 24,
              'uncontrolled_settle_s': .1,
              'status': meta['status'], 'contacts': meta['contacts'],
              'wall_s': time.perf_counter()-started,
              'dynamics_residual_N': meta['dynamics_residual_N'],
              'controls_finite': bool(np.all(np.isfinite(ctrl))),
              'legacy_two_palm_status': legacy_meta['status'],
              'legacy_controls_finite': bool(np.all(np.isfinite(legacy_ctrl)))}
except Exception as exc:
    report = {'scope': 'Cold QP solve only; no rollout, cloth or hand specialist',
              'four_palm_tasks': wbc.task_dim == 24,
              'uncontrolled_settle_s': .1,
              'status': 'failed', 'wall_s': time.perf_counter()-started,
              'error': f'{type(exc).__name__}: {exc}'}
(OUT / 'four_palm_wbc_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
if report['status'] == 'failed':
    raise RuntimeError(report['error'])
