"""Static thumb-side reachability from the joint-grasp rollout."""
from pathlib import Path
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares

root=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(root/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
trace=np.load(root/'joint_dressing_pilot_97ed662bb499d569_qpos.npz')['qpos']
data=mujoco.MjData(model)
thumb=model.body('g1_a/wuji_right/r_thumb_tip_sensor_frame').id
pad=model.body('dressing_pad').id
joint_ids=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/wuji_right/r_thumb_')]
qa=np.array([model.jnt_qposadr[j] for j in joint_ids])
lo=model.jnt_range[joint_ids,0]+1e-5;hi=model.jnt_range[joint_ids,1]-1e-5
rows=[]
for frame in (80,120,174):
 data.qpos[:]=trace[frame];mujoco.mj_forward(model,data)
 pad_center=data.xpos[pad].copy();start=data.qpos[qa].copy()
 for y,z in ((.035,0.),(.035,-.005),(.03,-.005),(.03,-.01)):
  target=pad_center+np.array([0.,y,z])
  def residual(x):
   data.qpos[qa]=x;mujoco.mj_forward(model,data)
   return data.xpos[thumb]-target
  fit=least_squares(residual,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=300)
  rows.append({'frame':frame,'target_yz_m':[y,z],
               'tip_error_m':float(np.linalg.norm(fit.fun)),
               'joint_target_rad':fit.x.tolist(),
               'joint_at_limit':bool(np.any(np.minimum(fit.x-lo,hi-fit.x)<1e-3))})
print(json.dumps(rows,indent=2))
