"""Static index-tip reach screen around a physically validated joint grasp state."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
state=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
base=state['qpos'].copy()
data.qpos[:]=base;mujoco.mj_forward(model,data)
pad=model.body('dressing_pad').id
tip=model.body('g1_a/wuji_right/r_index_finger_tip_sensor_frame').id
js=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/wuji_right/r_index_finger_')]
qa=np.array([model.jnt_qposadr[j] for j in js]);va=np.array([model.jnt_dofadr[j] for j in js])
q0=base[qa].copy()
lo=np.array([model.jnt_range[j,0]+.02 for j in js]);hi=np.array([model.jnt_range[j,1]-.02 for j in js])
def update(q):
    data.qpos[:]=base;data.qpos[qa]=q;mujoco.mj_forward(model,data)
results=[];states=[]
for x in (-.39,-.385,-.38):
    for y in (-.475,-.47,-.465):
        for z in (.694,.698,.702):
            goal=np.array([x,y,z])
            def residual(q):
                update(q)
                return np.r_[50*(data.xpos[tip]-goal),.04*(q-q0)]
            def jac(q):
                update(q);jp=np.zeros((3,model.nv))
                mujoco.mj_jac(model,data,jp,None,data.xpos[tip],tip)
                return np.vstack((50*jp[:,va],.04*np.eye(len(q))))
            opt=least_squares(residual,np.clip(q0,lo,hi),jac=jac,bounds=(lo,hi),max_nfev=80)
            update(opt.x)
            contacts=[];deep=[]
            for ci,c in enumerate(data.contact):
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    deep.append((bodies,float(c.dist)))
                if 'dressing_pad_contact' in geoms and any(b.startswith('g1_a/wuji_right/r_index_finger_') for b in bodies):
                    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                    contacts.append({'body':next(b for b in bodies if 'r_index_finger_' in b),
                                     'force_n':max(0.,float(wrench[0])),'distance_m':float(c.dist)})
            results.append({'goal_m':goal.tolist(),'tip_m':data.xpos[tip].tolist(),
                            'tip_error_m':float(np.linalg.norm(data.xpos[tip]-goal)),
                            'index_contacts':contacts,'deep_contact_count':len(deep),
                            'max_joint_change_rad':float(np.max(abs(opt.x-q0)))})
            states.append(data.qpos.copy())
ranked=sorted(range(len(results)),key=lambda i:(results[i]['deep_contact_count']>0,
            not any('distal' in c['body'] and c['force_n']<2. for c in results[i]['index_contacts']),
            results[i]['tip_error_m']))
for i in ranked[:8]:
    count=0;worst=0.
    for u in np.linspace(0.,1.,101):
        data.qpos[:]=(1-u)*base+u*states[i]
        mujoco.mj_forward(model,data)
        for c in data.contact:
            geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
            bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                    model.body(model.geom_bodyid[c.geom2]).name)
            if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                count+=1;worst=max(worst,-float(c.dist))
    results[i]['linear_path_deep_count']=count
    results[i]['linear_path_worst_penetration_m']=worst
summary={'scope':'Static index-tip reach only; dynamic pickup remains unverified',
         'screened':len(results),'base_tip_m':None,'best':[results[i] for i in ranked[:8]]}
data.qpos[:]=base;mujoco.mj_forward(model,data);summary['base_tip_m']=data.xpos[tip].tolist()
(OUT/'joint_dressing_index_tip_screen.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
np.savez_compressed(OUT/'joint_dressing_index_tip_candidates.npz',qpos=np.stack([states[i] for i in ranked[:8]]))
print(json.dumps(summary))
