"""Find A-right thumb contact opposing the index on the dressing pad."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'))
d=mujoco.MjData(m)
base=np.load(OUT/'joint_b_pinchtab_rrt_path.npz')['qpos'][-1].copy()
index_state=np.load(OUT/'joint_dressing_index_tip_candidates.npz')['qpos'][1]
d.qpos[:167]=base[:167]
for j in range(m.njnt):
    if m.joint(j).name.startswith('g1_a/wuji_right/r_index_finger_'):
        adr=int(m.jnt_qposadr[j]);d.qpos[adr]=index_state[adr]
base=d.qpos.copy();mujoco.mj_forward(m,d)
js=[j for j in range(m.njnt) if m.joint(j).name.startswith('g1_a/wuji_right/r_thumb_')]
qa=np.array([m.jnt_qposadr[j] for j in js]);va=np.array([m.jnt_dofadr[j] for j in js])
q0=base[qa].copy();lo=np.array([m.jnt_range[j,0]+.02 for j in js]);hi=np.array([m.jnt_range[j,1]-.02 for j in js])
tip=m.body('g1_a/wuji_right/r_thumb_tip_sensor_frame').id
def update(q):
    d.qpos[:]=base;d.qpos[qa]=q;mujoco.mj_forward(m,d)
def audit():
    normals={'thumb':np.zeros(3),'index':np.zeros(3)};counts={'thumb':0,'index':0};max_pen=0.
    for ci,c in enumerate(d.contact):
        geoms=[m.geom(int(g)).name for g in c.geom if g>=0]
        bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
        if 'floor' not in geoms and any(name.startswith(('g1_a/','g1_b/')) for name in bodies):max_pen=max(max_pen,max(0.,-float(c.dist)))
        if 'dressing_pad' not in bodies:continue
        for digit in normals:
            if any(name.startswith(f'g1_a/wuji_right/r_{digit}_') for name in bodies):
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                sign=1. if c.geom1>=0 and m.geom(c.geom1).name=='dressing_pad_contact' else -1.
                normals[digit]+=sign*max(0.,float(wrench[0]))*c.frame[:3];counts[digit]+=1
    denominator=float(np.linalg.norm(normals['thumb'])*np.linalg.norm(normals['index']))
    return counts,float(normals['thumb']@normals['index']/denominator) if denominator>1e-10 else None,max_pen
results=[];states=[]
for x in (-.405,-.399,-.393):
    for y in (-.409,-.404):
        for z in (.692,.696,.700):
            goal=np.array([x,y,z])
            def residual(q):
                update(q)
                return np.r_[40*(d.xpos[tip]-goal),.04*(q-q0)]
            def jac(q):
                update(q);jp=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,None,d.xpos[tip],tip)
                return np.vstack((40*jp[:,va],.04*np.eye(len(q))))
            fit=least_squares(residual,np.clip(q0,lo,hi),jac=jac,bounds=(lo,hi),max_nfev=80)
            update(fit.x);counts,cosine,pen=audit()
            results.append({'goal_m':goal.tolist(),'tip_m':d.xpos[tip].tolist(),
                            'tip_error_m':float(np.linalg.norm(d.xpos[tip]-goal)),
                            'max_joint_change_rad':float(np.max(abs(fit.x-q0))),
                            'contacts':counts,'normal_cosine':cosine,'max_nonfloor_penetration_m':pen,
                            'solver_success':bool(fit.success)})
            states.append(d.qpos.copy())
ranked=sorted(range(len(results)),key=lambda i:(results[i]['normal_cosine'] is None,
              results[i]['max_nonfloor_penetration_m']>.003,
              results[i]['normal_cosine'] if results[i]['normal_cosine'] is not None else 1.,
              results[i]['tip_error_m']))
report={'scope':'Static A-right pad side-contact search only; no pickup, force closure, or dynamic grasp',
        'screened':len(results),'best':[{'candidate_id':i,**results[i]} for i in ranked[:10]]}
(OUT/'joint_b_flex_a_thumb_side_contact_search.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(OUT/'joint_b_flex_a_thumb_side_contact_candidates.npz',qpos=np.stack([states[i] for i in ranked[:10]]))
print(json.dumps(report))
