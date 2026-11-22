"""Finite-difference audit of a dual-G1 flexible-strip contact Jacobian."""
import json
from pathlib import Path

import mujoco
import numpy as np

from flex_contact_jac import flex_contact_jacobian

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored.xml'))
data=mujoco.MjData(model)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
data.qpos[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
data.qvel[:]=np.random.default_rng(7).normal(0.,.01,model.nv)
mujoco.mj_forward(model,data)
contact=next(c for c in data.contact if 0 in c.flex and c.geom1>=0)
jac,_,verts,weights=flex_contact_jacobian(model,data,contact,1)
initial=weights@data.flexvert_xpos[verts]
trial=mujoco.MjData(model)
trial.qpos[:]=data.qpos
dt=1e-6
mujoco.mj_integratePos(model,trial.qpos,data.qvel,dt)
mujoco.mj_forward(model,trial)
finite_difference=(weights@trial.flexvert_xpos[verts]-initial)/dt
error=float(np.linalg.norm(finite_difference-jac@data.qvel))
report={'scope':'One flex material-point velocity Jacobian finite difference; no WBC rollout',
        'error_m_s':error,'finite_difference_m_s':finite_difference.tolist(),
        'jacobian_velocity_m_s':(jac@data.qvel).tolist(),'pass':error<1e-4}
(OUT/'joint_flex_contact_jacobian_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if not report['pass']:
    raise RuntimeError('Flex contact Jacobian finite difference failed')
