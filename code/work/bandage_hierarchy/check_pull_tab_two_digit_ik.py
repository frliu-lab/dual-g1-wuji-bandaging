"""Static arm/index reachability to the opposite side of the dressing pull tab."""
from pathlib import Path
import json,mujoco,numpy as np
from scipy.optimize import least_squares

root=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(root/'joint_spool_grasp_dressing_pull_tab_variant.xml'))
q=np.load(root/'joint_dressing_pilot_fbb7553db06c83e5_qpos.npz')['qpos'][-1]
d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
tab=m.geom('dressing_pull_tab').id
thumb=m.body('g1_a/wuji_right/r_thumb_tip_sensor_frame').id
index=m.body('g1_a/wuji_right/r_index_finger_tip_sensor_frame').id
wrist=m.body('g1_a/wuji_right/r_wrist').id
ids=[j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/right_') and
     any(s in m.joint(j).name for s in ('shoulder','elbow','wrist'))]
ids += [j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/wuji_right/r_index_finger_')]
qa=np.array([m.jnt_qposadr[j] for j in ids]);start=q[qa].copy()
lo=m.jnt_range[ids,0]+1e-5;hi=m.jnt_range[ids,1]-1e-5
tab_rot=d.geom_xmat[tab].reshape(3,3)
target=d.geom_xpos[tab]-tab_rot[:,1]*.007
thumb_start=d.xpos[thumb].copy();wrist_start=d.xpos[wrist].copy()
def residual(x):
 d.qpos[qa]=x;mujoco.mj_forward(m,d)
 return np.r_[d.xpos[index]-target,.4*(d.xpos[thumb]-thumb_start),
              .2*(d.xpos[wrist]-wrist_start),.002*(x[:7]-start[:7])]
fit=least_squares(residual,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=500)
print(json.dumps({'target_world_m':target.tolist(),
 'index_error_m':float(np.linalg.norm(fit.fun[:3])),
 'thumb_shift_m':float(np.linalg.norm(fit.fun[3:6])/.4),
 'wrist_shift_m':float(np.linalg.norm(fit.fun[6:9])/.2),
 'arm_delta_rad':{m.joint(j).name:float(v) for j,v in zip(ids[:7],fit.x[:7]-start[:7])},
 'index_target_rad':{m.joint(j).name:float(v) for j,v in zip(ids[7:],fit.x[7:])}},indent=2))
