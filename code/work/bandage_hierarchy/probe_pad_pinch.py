"""Joint arm/thumb IK for a two-tip dressing approach; geometric screen only."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
m = mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'))
d = mujoco.MjData(m)
base = np.load(OUT/'reach_ik_seed.npz')['qpos']
tips = [m.body('g1_a/wuji_right/'+n).id for n in
        ('r_index_finger_tip_sensor_frame','r_thumb_tip_sensor_frame')]
js = [j for j in range(m.njnt)
      if ((m.joint(j).name.startswith('g1_a/right_')
           and any(t in m.joint(j).name for t in ('shoulder','elbow','wrist')))
          or m.joint(j).name.startswith('g1_a/wuji_right/r_thumb_'))
      and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
qa = np.array([m.jnt_qposadr[j] for j in js], dtype=int)
va = np.array([m.jnt_dofadr[j] for j in js], dtype=int)
lo = np.array([m.jnt_range[j,0]+.03 for j in js])
hi = np.array([m.jnt_range[j,1]-.03 for j in js])
q0 = base[qa].copy()
results = []
candidate_q = []
for ty in (-.385, -.39, -.395, -.40):
    for tz in (.705, .71, .715, .72):
        goals = np.array([[-.38,-.44,.7164],[-.41,ty,tz]])
        def update(q):
            d.qpos[:] = base
            d.qpos[qa] = q
            mujoco.mj_forward(m,d)
        def residual(q):
            update(q)
            return np.r_[((d.xpos[tips]-goals)*30).ravel(), (q-q0)*.04]
        def jacobian(q):
            update(q)
            J = np.zeros((6+len(q),len(q)))
            for k,bid in enumerate(tips):
                jp = np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jp,None,d.xpos[bid],bid)
                J[k*3:k*3+3] = 30*jp[:,va]
            J[6:] = .04*np.eye(len(q))
            return J
        opt = least_squares(residual,q0,jac=jacobian,bounds=(lo,hi),max_nfev=100,
                            ftol=1e-9,xtol=1e-9,gtol=1e-9)
        update(opt.x)
        deep=[]
        pad=[]
        for i,c in enumerate(d.contact):
            if c.geom1 < 0 or c.geom2 < 0:continue
            names=[m.geom(g).name for g in (c.geom1,c.geom2)]
            bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
            if 'floor' in names or not any(b.startswith(('g1_a/','g1_b/')) for b in bodies):continue
            if c.dist < -.003:deep.append({'bodies':bodies,'distance_m':float(c.dist)})
            if 'dressing_pad_contact' in names:
                wrench=np.zeros(6)
                mujoco.mj_contactForce(m,d,i,wrench)
                pad.append({'bodies':bodies,'distance_m':float(c.dist),
                            'normal_force_n':float(wrench[0])})
        results.append({'goals_m':goals.tolist(), 'tips_m':d.xpos[tips].tolist(),
                        'errors_m':np.linalg.norm(d.xpos[tips]-goals,axis=1).tolist(),
                        'joint_change_rad':float(np.max(np.abs(opt.x-q0))),
                        'deep_contacts':deep,'pad_contacts':pad,'success':bool(opt.success)})
        candidate_q.append(d.qpos.copy())
eligible=[i for i,r in enumerate(results)
          if not r['deep_contacts'] and max(r['errors_m']) < .005
          and any('r_thumb_' in b for c in r['pad_contacts'] for b in c['bodies'])
          and any('r_index_' in b for c in r['pad_contacts'] for b in c['bodies'])]
selected=min(eligible,key=lambda i:sum(results[i]['errors_m'])+.002*results[i]['joint_change_rad']) if eligible else None
path_bad=[]
path_trials={}
if selected is not None:
    qend=candidate_q[selected]
    arm_qa=qa[:7]
    thumb_qa=qa[7:]
    arm_first=base.copy();arm_first[arm_qa]=qend[arm_qa]
    thumb_first=base.copy();thumb_first[thumb_qa]=qend[thumb_qa]
    paths={'simultaneous':[base,qend],
           'arm_then_thumb':[base,arm_first,qend],
           'thumb_then_arm':[base,thumb_first,qend]}
    for strategy,keyframes in paths.items():
        trial=[]
        for segment,(start,end) in enumerate(zip(keyframes,keyframes[1:])):
            for u in np.linspace(0,1,101):
                d.qpos[:]=(1-u)*start+u*end
                mujoco.mj_forward(m,d)
                for c in d.contact:
                    if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
                    names=[m.geom(g).name for g in (c.geom1,c.geom2)]
                    bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
                    if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                        trial.append({'segment':segment,'u':float(u),
                                      'bodies':bodies,'distance_m':float(c.dist)})
        path_trials[strategy]={'deep_contacts':len(trial),
                               'worst_contacts':sorted(trial,key=lambda x:x['distance_m'])[:5]}
    path_bad=path_trials['simultaneous']['worst_contacts']
    np.savez_compressed(OUT/'pad_pinch_seed.npz',qpos=qend,initial_qpos=base)
report={'scope':'Static combined seven-arm-joint and four-thumb-joint two-tip IK; no grasp or WBC execution',
        'candidates':results,'selected_candidate_index':selected,
        'path_trials':path_trials}
(OUT/'pad_pinch_scan.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps([{'thumb_goal':r['goals_m'][1],
                   'errors_mm':[round(e*1000,1) for e in r['errors_m']],
                   'deep':len(r['deep_contacts']),'pad':len(r['pad_contacts'])}
                  for r in results]))
