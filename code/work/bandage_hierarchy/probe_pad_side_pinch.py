"""Search lateral thumb/index contacts on opposite dressing-pad edges."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/wuji_multiagent/bandage_v0'
scene = ET.parse(OUT/'scene.xml')
pad_geom = next(g for g in scene.getroot().iter('geom') if g.get('name') == 'dressing_pad_contact')
pad_geom.set('size', '.045 .03 .008')
stage_geom = next(g for g in scene.getroot().iter('geom') if g.get('name') == 'dressing_stage')
stage_geom.set('size', '.04 .015 .006')
m = mujoco.MjModel.from_xml_string(ET.tostring(scene.getroot(),encoding='unicode'))
d = mujoco.MjData(m)
base = np.load(OUT/'reach_ik_seed.npz')['qpos']
tip_names = ('g1_a/wuji_right/r_index_finger_tip_sensor_frame',
             'g1_a/wuji_right/r_thumb_tip_sensor_frame')
tips = [m.body(n).id for n in tip_names]
js = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and
      ((m.joint(j).name.startswith('g1_a/right_') and
        any(t in m.joint(j).name for t in ('shoulder','elbow','wrist')))
       or m.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_',
                                      'g1_a/wuji_right/r_index_finger_')))]
qa = np.array([m.jnt_qposadr[j] for j in js],dtype=int)
va = np.array([m.jnt_dofadr[j] for j in js],dtype=int)
q0=base[qa].copy()
lo=np.array([m.jnt_range[j,0]+.03 for j in js])
hi=np.array([m.jnt_range[j,1]-.03 for j in js])
def update(q):
    d.qpos[:]=base
    d.qpos[qa]=q
    mujoco.mj_forward(m,d)
results=[]
seeds=[]
for iy in (-.475,-.48,-.485):
    for ty in (-.405,-.41,-.415):
        for z in (.694,.698,.702):
            goals=np.array([[-.38,iy,z],[-.41,ty,z]])
            def residual(q):
                update(q)
                return np.r_[30*(d.xpos[tips]-goals).ravel(),.03*(q-q0)]
            def jacobian(q):
                update(q)
                J=np.zeros((6+len(q),len(q)))
                for k,bid in enumerate(tips):
                    jp=np.zeros((3,m.nv))
                    mujoco.mj_jac(m,d,jp,None,d.xpos[bid],bid)
                    J[3*k:3*k+3]=30*jp[:,va]
                J[6:]=.03*np.eye(len(q))
                return J
            opt=least_squares(residual,q0,jac=jacobian,bounds=(lo,hi),max_nfev=120)
            update(opt.x)
            deep=[]
            contacts={'index':[],'thumb':[]}
            normals={'index':np.zeros(3),'thumb':np.zeros(3)}
            for i,c in enumerate(d.contact):
                if c.geom1<0 or c.geom2<0:continue
                names=[m.geom(g).name for g in (c.geom1,c.geom2)]
                bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
                if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
                    deep.append({'bodies':bodies,'distance_m':float(c.dist)})
                if 'dressing_pad_contact' not in names:continue
                for digit in contacts:
                    if any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                        wrench=np.zeros(6)
                        mujoco.mj_contactForce(m,d,i,wrench)
                        force=max(0.,float(wrench[0]))
                        contacts[digit].append({'body':next(b for b in bodies if b.startswith('g1_a/')),
                                                'distance_m':float(c.dist),'normal_force_n':force})
                        sign=1 if names[0]=='dressing_pad_contact' else -1
                        normals[digit]+=sign*force*np.asarray(c.frame[:3])
            a,b=normals.values()
            cosine=(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                    if np.linalg.norm(a)*np.linalg.norm(b)>1e-8 else None)
            results.append({'goals_m':goals.tolist(),
                            'tip_errors_m':np.linalg.norm(d.xpos[tips]-goals,axis=1).tolist(),
                            'max_joint_change_rad':float(np.max(abs(opt.x-q0))),
                            'deep_contacts':deep,'pad_contacts':contacts,
                            'contact_normal_cosine':cosine,'solver_success':bool(opt.success)})
            seeds.append(opt.x.copy())
ranked=sorted((i for i,r in enumerate(results) if r['contact_normal_cosine'] is not None
               and r['contact_normal_cosine']<-.5),
              key=lambda i:(sum(max(0.,-x['distance_m']-.003)**2
                                for x in results[i]['deep_contacts']),
                            sum(results[i]['tip_errors_m'])))[:4]
refined=[]
refined_q=[]
for i in ranked:
    goals=np.asarray(results[i]['goals_m'])
    def objective(q):
        update(q)
        tip_loss=1e4*np.sum((d.xpos[tips]-goals)**2)
        collision_loss=0.
        for c in d.contact:
            if c.geom1<0 or c.geom2<0:continue
            names=[m.geom(g).name for g in (c.geom1,c.geom2)]
            bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
            if 'floor' in names or not any(b.startswith(('g1_a/','g1_b/')) for b in bodies):continue
            collision_loss+=max(0.,-float(c.dist)-.002)**2
        return float(tip_loss+2e5*collision_loss+.002*np.sum((q-q0)**2))
    opt=minimize(objective,seeds[i],method='L-BFGS-B',bounds=list(zip(lo,hi)),
                 options={'maxiter':80,'ftol':1e-8})
    update(opt.x)
    deep=[];normals={'thumb':np.zeros(3),'index':np.zeros(3)};forces={'thumb':0.,'index':0.}
    for ci,c in enumerate(d.contact):
        if c.geom1<0 or c.geom2<0:continue
        names=[m.geom(g).name for g in (c.geom1,c.geom2)]
        bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
        if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
            deep.append({'bodies':bodies,'distance_m':float(c.dist)})
        if 'dressing_pad_contact' not in names:continue
        for digit in normals:
            if any(b.startswith(f'g1_a/wuji_right/r_{digit}_') for b in bodies):
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                force=max(0.,float(wrench[0]));forces[digit]+=force
                sign=1 if names[0]=='dressing_pad_contact' else -1
                normals[digit]+=sign*force*np.asarray(c.frame[:3])
    a,b=normals.values()
    cos=(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
         if np.linalg.norm(a)*np.linalg.norm(b)>1e-8 else None)
    refined.append({'source_candidate_index':i,'goals_m':goals.tolist(),
                    'tip_errors_m':np.linalg.norm(d.xpos[tips]-goals,axis=1).tolist(),
                    'deep_contacts':deep,'forces_n':forces,'contact_normal_cosine':cos,
                    'solver_success':bool(opt.success),'solver_iterations':int(opt.nit)})
    refined_q.append(d.qpos.copy())
for item,qend in zip(refined,refined_q):
    finger_first=base.copy();finger_first[qa[7:]]=qend[qa[7:]]
    arm_first=base.copy();arm_first[qa[:7]]=qend[qa[:7]]
    paths={'simultaneous':[base,qend],
           'finger_then_arm':[base,finger_first,qend],
           'arm_then_finger':[base,arm_first,qend]}
    item['path_deep_counts']={}
    for name,keys in paths.items():
        count=0;worst=0.
        for start,end in zip(keys,keys[1:]):
            for u in np.linspace(0,1,101):
                d.qpos[:]=(1-u)*start+u*end
                mujoco.mj_forward(m,d)
                for c in d.contact:
                    if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
                    names=[m.geom(g).name for g in (c.geom1,c.geom2)]
                    bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
                    if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                        count+=1;worst=max(worst,-float(c.dist))
        item['path_deep_counts'][name]={'count':count,'worst_penetration_m':worst}
report={'scope':'Static 15-DOF side-pinch screen on an isolated smaller dressing-pad variant; no dynamic grasp or object lift',
        'pad_half_size_m':[.045,.03,.008],
        'stage_half_size_m':[.04,.015,.006],
        'candidates':results,'collision_refined_candidates':refined,
        'safe_opposed_refined_terminal_count':sum(not r['deep_contacts'] and r['contact_normal_cosine'] is not None
                                                  and r['contact_normal_cosine']<-.5 for r in refined),
        'safe_opposed_refined_path_count':sum(not r['deep_contacts'] and r['contact_normal_cosine'] is not None
                                              and r['contact_normal_cosine']<-.5
                                              and any(p['count']==0 for p in r['path_deep_counts'].values())
                                              for r in refined)}
(OUT/'pad_side_pinch_scan.json').write_text(json.dumps(report,indent=2),encoding='utf8')
best=sorted(results,key=lambda r:(len(r['deep_contacts'])>0,
                                  r['contact_normal_cosine'] is None or r['contact_normal_cosine']>-.5,
                                  sum(r['tip_errors_m'])))[:8]
print(json.dumps([{'goals':r['goals_m'],'errors_mm':[round(v*1000,1) for v in r['tip_errors_m']],
                   'deep':len(r['deep_contacts']),
                   'forces_n':{k:round(sum(x['normal_force_n'] for x in v),3) for k,v in r['pad_contacts'].items()},
                   'normal_cos':r['contact_normal_cosine']} for r in best]))
print(json.dumps({'refined':[{'source':r['source_candidate_index'],
                              'errors_mm':[round(v*1000,1) for v in r['tip_errors_m']],
                              'deep':len(r['deep_contacts']),'forces_n':r['forces_n'],
                              'normal_cos':r['contact_normal_cosine']}
                             for r in refined]}))
