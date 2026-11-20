"""Static index-finger reachability at saved joint-pickup frames (diagnostic only)."""
from pathlib import Path
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
MODEL=OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'
TRACE=OUT/'joint_dressing_pilot_3c5cfdde0ecdd317_qpos.npz'
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m)
q=np.load(TRACE)['qpos']
tip=m.body('g1_a/wuji_right/r_index_finger_tip_sensor_frame').id
pad=m.body('dressing_pad').id
joint_ids=[j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/wuji_right/r_index_finger_')]
qa=np.array([m.jnt_qposadr[j] for j in joint_ids])
lo=m.jnt_range[joint_ids,0]+1e-5;hi=m.jnt_range[joint_ids,1]-1e-5
rows=[]
for frame in (40,80,120,174):
 d.qpos[:]=q[frame];mujoco.mj_forward(m,d)
 target=d.xpos[pad].copy()+np.array([0.,-.02,.002])
 initial=d.xpos[tip].copy();q0=d.qpos[qa].copy()
 def residual(x):
  d.qpos[qa]=x;mujoco.mj_forward(m,d)
  return d.xpos[tip]-target
 fit=least_squares(residual,np.clip(q0,lo,hi),bounds=(lo,hi),max_nfev=250,
                   xtol=1e-10,ftol=1e-10,gtol=1e-10)
 rows.append({'frame':frame,'time_s':float((frame+1)*.02),
  'initial_tip_target_distance_m':float(np.linalg.norm(initial-target)),
  'best_index_only_distance_m':float(np.linalg.norm(fit.fun)),
  'joint_at_limit':bool(np.any(np.minimum(fit.x-lo,hi-fit.x)<1e-3)),
  'optimized_joint_delta_rad':(fit.x-q0).tolist()})
 if frame in (80,174):
  arm_ids=[j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/right_')
           and any(s in m.joint(j).name for s in ('shoulder','elbow','wrist'))]
  arm_qa=np.array([m.jnt_qposadr[j] for j in arm_ids]);all_qa=np.r_[arm_qa,qa]
  arm_start=q[frame,arm_qa].copy();start=np.r_[arm_start,q0]
  all_lo=m.jnt_range[np.r_[arm_ids,joint_ids],0]+1e-5
  all_hi=m.jnt_range[np.r_[arm_ids,joint_ids],1]-1e-5
  d.qpos[:]=q[frame];mujoco.mj_forward(m,d);palm=m.body('g1_a/wuji_right/r_wrist').id
  palm_start=d.xpos[palm].copy()
  def arm_residual(x):
   d.qpos[all_qa]=x;mujoco.mj_forward(m,d)
   return np.r_[d.xpos[tip]-target, .2*(d.xpos[palm]-palm_start),
                .002*(x[:len(arm_qa)]-arm_start)]
  joint_fit=least_squares(arm_residual,np.clip(start,all_lo,all_hi),bounds=(all_lo,all_hi),max_nfev=350)
  rows[-1]['arm_plus_index']={'tip_target_distance_m':float(np.linalg.norm(joint_fit.fun[:3])),
      'wrist_translation_m':float(np.linalg.norm(joint_fit.fun[3:6])/.2),
      'max_arm_joint_delta_rad':float(np.max(np.abs(joint_fit.x[:len(arm_qa)]-arm_start)))}
  rows[-1]['arm_plus_index']['arm_joint_delta_rad']={m.joint(j).name:float(v)
      for j,v in zip(arm_ids,joint_fit.x[:len(arm_qa)]-arm_start)}
  rows[-1]['arm_plus_index']['index_joint_target_rad']={m.joint(j).name:float(v)
      for j,v in zip(joint_ids,joint_fit.x[len(arm_qa):])}
print(json.dumps({'scope':'Static kinematic diagnostic; does not prove dynamic feasibility',
                  'target':'Pad center plus 20 mm negative Y and 2 mm positive Z',
                  'frames':rows},indent=2))
