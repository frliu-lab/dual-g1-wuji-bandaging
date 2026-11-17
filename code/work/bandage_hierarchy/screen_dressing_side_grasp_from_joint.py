"""Search side-facing A-right thumb/index fingertip poses at the joint grasp endpoint."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser();parser.add_argument('--refine',action='store_true');args=parser.parse_args()
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
base=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')['qpos']
tip_ids=[model.body(n).id for n in ('g1_a/wuji_right/r_thumb_tip_sensor_frame',
                                    'g1_a/wuji_right/r_index_finger_tip_sensor_frame')]
js=[j for j in range(model.njnt) if model.joint(j).name.startswith(
    ('g1_a/wuji_right/r_thumb_','g1_a/wuji_right/r_index_finger_'))]
qa=np.array([model.jnt_qposadr[j] for j in js]);va=np.array([model.jnt_dofadr[j] for j in js])
q0=base[qa]
lo=np.array([model.jnt_range[j,0]+.02 for j in js]);hi=np.array([model.jnt_range[j,1]-.02 for j in js])
def update(q):
    data.qpos[:]=base;data.qpos[qa]=q;mujoco.mj_forward(model,data)
rows=[];states=[]
for thumb_y in (np.linspace(-.405,-.400,11) if args.refine else (-.41,-.405,-.40)):
    for index_y in ((-.475,) if args.refine else (-.475,-.47,-.465)):
        for z in ((.698,.700,.702) if args.refine else (.692,.696,.700)):
            goals=np.array([[-.408,thumb_y,z],[-.385,index_y,z]])
            def residual(q):
                update(q)
                return np.r_[50*(data.xpos[tip_ids]-goals).ravel(),.04*(q-q0)]
            def jac(q):
                update(q);J=np.zeros((6+len(q),len(q)))
                for k,bid in enumerate(tip_ids):
                    jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,data.xpos[bid],bid)
                    J[3*k:3*k+3]=50*jp[:,va]
                J[6:]=.04*np.eye(len(q))
                return J
            opt=least_squares(residual,np.clip(q0,lo,hi),jac=jac,bounds=(lo,hi),max_nfev=100)
            update(opt.x)
            forces={'thumb':0.,'index':0.};normals={'thumb':np.zeros(3),'index':np.zeros(3)}
            deep=[];parts={'thumb':set(),'index':set()}
            for ci,c in enumerate(data.contact):
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    deep.append({'bodies':bodies,'distance_m':float(c.dist)})
                if 'dressing_pad_contact' not in geoms:continue
                for digit,label in (('thumb','thumb'),('index','index_finger')):
                    body=next((b for b in bodies if b.startswith(f'g1_a/wuji_right/r_{label}_')),None)
                    if body:
                        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                        force=max(0.,float(wrench[0]));forces[digit]+=force;parts[digit].add(body)
                        sign=1. if geoms[0]=='dressing_pad_contact' else -1.
                        normals[digit]+=sign*force*np.asarray(c.frame[:3])
            a,b=normals.values()
            cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>1e-8 else None
            rows.append({'goals_m':goals.tolist(),'tips_m':data.xpos[tip_ids].tolist(),
                         'tip_errors_m':np.linalg.norm(data.xpos[tip_ids]-goals,axis=1).tolist(),
                         'forces_n':forces,'normal_cosine':cos,
                         'normal_directions':{k:(v/np.linalg.norm(v)).tolist() if np.linalg.norm(v)>0 else None for k,v in normals.items()},
                         'contact_bodies':{k:sorted(v) for k,v in parts.items()},
                         'deep_contacts':deep,'max_joint_change_rad':float(np.max(abs(opt.x-q0)))})
            states.append(data.qpos.copy())
ranked=sorted(range(len(rows)),key=lambda i:(bool(rows[i]['deep_contacts']),
    rows[i]['normal_cosine'] is None or rows[i]['normal_cosine']>-.5,
    not all(.05<=f<=2. for f in rows[i]['forces_n'].values()),
    sum(rows[i]['tip_errors_m'])))
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
    rows[i]['linear_path_deep_count']=count
    rows[i]['linear_path_worst_penetration_m']=worst
summary={'scope':'Static side-grasp screen from verified joint endpoint; no dynamic pickup claim',
         'screened':len(rows),'strict_static_count':sum(not r['deep_contacts'] and r['normal_cosine'] is not None
             and r['normal_cosine']<-.5 and all(.05<=f<=2. for f in r['forces_n'].values()) for r in rows),
         'best':[rows[i] for i in ranked[:8]]}
suffix='_refined' if args.refine else ''
(OUT/f'joint_dressing_side_grasp_from_joint_screen{suffix}.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
np.savez_compressed(OUT/f'joint_dressing_side_grasp_from_joint_candidates{suffix}.npz',qpos=np.stack([states[i] for i in ranked[:8]]))
print(json.dumps({'screened':summary['screened'],'strict_static_count':summary['strict_static_count'],
                  'best':[{'errors_mm':[round(x*1000,1) for x in r['tip_errors_m']],
                           'forces_n':r['forces_n'],'cos':r['normal_cosine'],
                           'normals':r['normal_directions'],'deep':len(r['deep_contacts'])} for r in summary['best'][:4]]}))
