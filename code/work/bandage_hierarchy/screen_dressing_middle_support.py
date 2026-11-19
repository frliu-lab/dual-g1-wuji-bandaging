"""Check whether A-right middle finger can join the side pinch without collision."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
base=np.load(OUT/'joint_dressing_side_grasp_from_joint_candidates_refined.npz')['qpos'][0]
body=model.body('g1_a/wuji_right/r_middle_finger_tip_sensor_frame').id
js=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_a/wuji_right/r_middle_finger_')]
qa=np.array([model.jnt_qposadr[j] for j in js]);va=np.array([model.jnt_dofadr[j] for j in js])
lo=np.array([model.jnt_range[j,0]+.02 for j in js]);hi=np.array([model.jnt_range[j,1]-.02 for j in js])
def update(q):
    data.qpos[:]=base;data.qpos[qa]=q;mujoco.mj_forward(model,data)
rows=[];states=[]
for x in (-.415,-.405,-.395,-.385):
    for y in (-.48,-.475,-.47):
        for z in (.695,.705,.715,.725):
            goal=np.array([x,y,z])
            def residual(q):
                update(q);return np.r_[50*(data.xpos[body]-goal),.03*(q-base[qa])]
            def jac(q):
                update(q);jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,data.xpos[body],body)
                return np.vstack([50*jp[:,va],.03*np.eye(len(q))])
            opt=least_squares(residual,np.clip(base[qa],lo,hi),jac=jac,bounds=(lo,hi),max_nfev=100)
            update(opt.x)
            force=0.;parts=[];deep=[]
            for ci,c in enumerate(data.contact):
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    deep.append({'bodies':bodies,'distance_m':float(c.dist)})
                if 'dressing_pad_contact' in geoms and any(b.startswith('g1_a/wuji_right/r_middle_finger_') for b in bodies):
                    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                    force+=max(0.,float(wrench[0]));parts.append(bodies)
            rows.append({'goal_m':goal.tolist(),'tip_m':data.xpos[body].tolist(),
                         'tip_error_m':float(np.linalg.norm(data.xpos[body]-goal)),
                         'middle_force_n':force,'contacts':parts,'deep_contacts':deep,
                         'joint_change_max_rad':float(np.max(abs(opt.x-base[qa])))})
            states.append(data.qpos.copy())
ranked=sorted(range(len(rows)),key=lambda i:(bool(rows[i]['deep_contacts']),
                                                 not .05<=rows[i]['middle_force_n']<=2.,
                                                 rows[i]['tip_error_m']))
summary={'scope':'Static middle-finger addition to A-right thumb/index side pinch; dynamic hold not validated',
         'screened':len(rows),'collision_free_contact_count':sum(not r['deep_contacts'] and .05<=r['middle_force_n']<=2. for r in rows),
         'best':[rows[i] for i in ranked[:8]]}
(OUT/'joint_dressing_middle_support_screen.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
np.savez_compressed(OUT/'joint_dressing_middle_support_candidates.npz',qpos=np.stack([states[i] for i in ranked[:8]]))
print(json.dumps({'screened':summary['screened'],'collision_free_contact_count':summary['collision_free_contact_count'],
                  'best':[{'goal':r['goal_m'],'error_mm':round(r['tip_error_m']*1000,1),
                           'force':round(r['middle_force_n'],3),'deep':len(r['deep_contacts'])} for r in summary['best'][:4]]}))
