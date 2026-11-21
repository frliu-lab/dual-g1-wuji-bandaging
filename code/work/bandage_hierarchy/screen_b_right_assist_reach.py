"""Static reach screen for the fourth hand near the training forearm."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
scene=OUT/'joint_spool_grasp_dressing_pinch_post_variant.xml'
state=OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz'
model=mujoco.MjModel.from_xml_path(str(scene))
data=mujoco.MjData(model)
data.qpos[:]=np.load(state)['qpos']
mujoco.mj_forward(model,data)
wrist=model.body('g1_b/wuji_right/r_wrist').id
forearm=model.body('training_forearm_dynamic').id
arm_ids=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/right_')
         and any(token in model.joint(j).name for token in ('shoulder','elbow','wrist'))]
qa=np.array([model.jnt_qposadr[j] for j in arm_ids])
start=data.qpos[qa].copy()
lo=model.jnt_range[arm_ids,0]+1e-5
hi=model.jnt_range[arm_ids,1]-1e-5
forearm_pos=data.xpos[forearm].copy()
spool_pos=data.xpos[model.body('bandage_spool').id].copy()
initial_wrist_to_forearm=float(np.linalg.norm(data.xpos[wrist]-forearm_pos))
target_specs=[('forearm',forearm_pos,offset) for offset in
              ((.10,.00,-.02),(.14,-.05,-.04),(.18,.00,-.02),(.22,-.05,-.04))]
target_specs += [('spool',spool_pos,offset) for offset in
                 ((0.,.20,.10),(0.,.25,.10),(.10,.20,.10),(.10,.25,.10))]
results=[]
seeds=[start,(lo+hi)/2,lo+.25*(hi-lo),lo+.75*(hi-lo)]
for reference,origin,offset in target_specs:
    goal=origin+np.array(offset)
    def residual(x):
        data.qpos[qa]=x
        mujoco.mj_forward(model,data)
        return np.r_[data.xpos[wrist]-goal,.005*(x-start)]
    fits=[least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=300)
          for seed in seeds]
    fit=min(fits,key=lambda result:np.linalg.norm(result.fun[:3]))
    residual(fit.x)
    deep_contacts=[{'bodies':[model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)],
                    'distance_m':float(c.dist)} for c in data.contact
                   if c.dist<-.003 and any(model.body(model.geom_bodyid[g]).name.startswith('g1_b/wuji_right/')
                                             for g in (c.geom1,c.geom2))]
    results.append({'reference':reference,'target_offset_from_reference_m':offset,
                    'target_world_m':goal.tolist(),
                    'starts_screened':len(seeds),
                    'wrist_error_m':float(np.linalg.norm(data.xpos[wrist]-goal)),
                    'arm_delta_norm_rad':float(np.linalg.norm(fit.x-start)),
                    'deep_b_right_contacts':deep_contacts,'joint_targets_rad':fit.x.tolist()})
report={'scope':'Static fourth-hand wrist reach only; no finger grasp, trajectory, or dynamics pass',
        'scene':scene.name,'joint_names':[model.joint(j).name for j in arm_ids],
        'checkpoint_wrist_to_forearm_m':initial_wrist_to_forearm,
        'results':results}
name='joint_b_right_assist_static_reach_audit.json'
(OUT/name).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({'output':name,'results':[{k:v for k,v in item.items() if k not in ('joint_targets_rad','deep_b_right_contacts')}
                                           | {'deep_contacts':len(item['deep_b_right_contacts'])} for item in results]}))
