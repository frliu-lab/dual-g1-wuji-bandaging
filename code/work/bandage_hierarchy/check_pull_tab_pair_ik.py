"""Static two-finger side-pinch IK on the physical pull tab (diagnostic)."""
from pathlib import Path
import json,mujoco,numpy as np
from scipy.optimize import least_squares

root=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(root/'joint_spool_grasp_dressing_pull_tab_variant.xml'))
q=np.load(root/'joint_dressing_pilot_687146f9fc5ab552_qpos.npz')['qpos'][-1]
d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
tab=m.geom('dressing_pull_tab').id;R=d.geom_xmat[tab].reshape(3,3);center=d.geom_xpos[tab].copy()
targets={'thumb':center+R[:,1]*.008,'index':center-R[:,1]*.008}
bodies={name:m.body(f'g1_a/wuji_right/r_{name}_finger_tip_sensor_frame' if name=='index' else
                     'g1_a/wuji_right/r_thumb_tip_sensor_frame').id for name in targets}
ids=[j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/right_') and
     any(s in m.joint(j).name for s in ('shoulder','elbow','wrist'))]
for prefix in ('r_thumb_','r_index_finger_'):
 ids += [j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/wuji_right/'+prefix)]
qa=np.array([m.jnt_qposadr[j] for j in ids]);start=q[qa].copy()
lo=m.jnt_range[ids,0]+1e-5;hi=m.jnt_range[ids,1]-1e-5
wrist=m.body('g1_a/wuji_right/r_wrist').id;wrist_start=d.xpos[wrist].copy()
def residual(x):
 d.qpos[qa]=x;mujoco.mj_forward(m,d)
 return np.r_[d.xpos[bodies['thumb']]-targets['thumb'],
              d.xpos[bodies['index']]-targets['index'],
              .2*(d.xpos[wrist]-wrist_start),.002*(x[:7]-start[:7])]
fit=least_squares(residual,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=500)
print(json.dumps({'target_thumb_m':targets['thumb'].tolist(),'target_index_m':targets['index'].tolist(),
 'thumb_error_m':float(np.linalg.norm(fit.fun[:3])),'index_error_m':float(np.linalg.norm(fit.fun[3:6])),
 'wrist_shift_m':float(np.linalg.norm(fit.fun[6:9])/.2),
 'arm_delta_rad':{m.joint(j).name:float(v) for j,v in zip(ids[:7],fit.x[:7]-start[:7])},
 'thumb_target_rad':{m.joint(j).name:float(v) for j,v in zip(ids[7:11],fit.x[7:11])},
 'index_target_rad':{m.joint(j).name:float(v) for j,v in zip(ids[11:],fit.x[11:])}},indent=2))
