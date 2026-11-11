"""Find an open, contact-free two-finger pose near the composite pinch endpoint."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_support_pinch_variant.xml'))
data=mujoco.MjData(model)
pinch=np.load(OUT/'joint_support_pinch_static_candidate.npz')['qpos'].copy()
tip_names=('g1_a/wuji_right/r_index_finger_tip_sensor_frame',
           'g1_a/wuji_right/r_thumb_tip_sensor_frame')
tips=[model.body(n).id for n in tip_names]
js=[j for j in range(model.njnt) if model.joint(j).name.startswith(
    ('g1_a/wuji_right/r_index_finger_','g1_a/wuji_right/r_thumb_'))]
qa=np.array([model.jnt_qposadr[j] for j in js],dtype=int)
va=np.array([model.jnt_dofadr[j] for j in js],dtype=int)
lo=np.array([model.jnt_range[j,0]+.03 for j in js]);hi=np.array([model.jnt_range[j,1]-.03 for j in js])
data.qpos[:]=pinch;mujoco.mj_forward(model,data)
tip_start=data.xpos[tips].copy()
results=[]
for index_retreat,thumb_retreat in ((.015,.005),(.015,.0075),(.015,.01),(.01,.01),(.015,.015),(.02,.02)):
    goals=tip_start.copy();goals[0,1]-=index_retreat;goals[1,1]+=thumb_retreat
    def forward(x):
        data.qpos[:]=pinch;data.qpos[qa]=x;mujoco.mj_forward(model,data)
    def residual(x):
        forward(x)
        return np.r_[30*(data.xpos[tips]-goals).ravel(),.02*(x-pinch[qa])]
    def jacobian(x):
        forward(x);J=np.zeros((6+len(x),len(x)))
        for i,bid in enumerate(tips):
            jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,data.xpos[bid],bid)
            J[3*i:3*i+3]=30*jp[:,va]
        J[6:]=.02*np.eye(len(x));return J
    fit=least_squares(residual,np.clip(pinch[qa],lo+1e-5,hi-1e-5),jac=jacobian,
                      bounds=(lo,hi),max_nfev=100)
    forward(fit.x)
    pad_contacts=[];deep=[]
    for c in data.contact:
        if c.geom1<0 or c.geom2<0:continue
        geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
        bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                model.body(model.geom_bodyid[c.geom2]).name)
        if 'dressing_pad_contact' in geoms and any(b.startswith('g1_a/wuji_right/') for b in bodies):
            pad_contacts.append({'bodies':bodies,'distance_m':float(c.dist)})
        if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
            deep.append({'bodies':bodies,'distance_m':float(c.dist)})
    eligible=not pad_contacts and not deep
    row={'index_retreat_m':index_retreat,'thumb_retreat_m':thumb_retreat,
         'tip_errors_m':np.linalg.norm(data.xpos[tips]-goals,axis=1).tolist(),
         'pad_contacts':pad_contacts,'deep_contacts':deep,'solver_success':bool(fit.success),
         'eligible_pregrasp':eligible}
    results.append(row)
    if eligible:
        np.savez_compressed(OUT/'joint_support_pinch_pregrasp.npz',qpos=data.qpos.copy())
        pregrasp=data.qpos.copy()
        closure_deep=[]
        closure_first_contact={'thumb':None,'index':None}
        for u in np.linspace(0,1,101):
            data.qpos[:]=(1-u)*pregrasp+u*pinch
            mujoco.mj_forward(model,data)
            for c in data.contact:
                if c.geom1<0 or c.geom2<0:continue
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    if len(closure_deep)<5:closure_deep.append({'fraction':float(u),'bodies':bodies,
                                                                 'distance_m':float(c.dist)})
                if 'dressing_pad_contact' in geoms:
                    for digit in closure_first_contact:
                        if closure_first_contact[digit] is None and any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                            closure_first_contact[digit]=float(u)
        row['closure_deep_contacts']=closure_deep
        row['closure_first_contact_fraction']=closure_first_contact
        row['closure_collision_free']=not closure_deep
        break
report={'scope':'Static pregrasp finger opening at the composite endpoint; no approach path or dynamic claim',
        'results':results}
(OUT/'joint_support_pinch_pregrasp_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'results':[{k:v for k,v in r.items() if k not in ('pad_contacts','deep_contacts')}
                              | {'pad_contact_count':len(r['pad_contacts']),'deep_count':len(r['deep_contacts'])}
                              for r in results]}))
