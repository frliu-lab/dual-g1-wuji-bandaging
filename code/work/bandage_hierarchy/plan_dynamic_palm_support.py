"""Collision-screened RRT path to the isolated palm-only support pose."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
stem='dynamic_palm_support_search_root_m0p3_pinky0p5'
candidate=np.load(OUT/f'{stem}_candidate.npz')['qpos'].copy()
scene=ET.parse(OUT/'dynamic_forearm_variant.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos','-.3 0 .8')
model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
data=mujoco.MjData(model)
report_search=json.loads((OUT/f'{stem}.json').read_text(encoding='utf8'))
names=report_search['candidate']['joint_names']
qa=np.array([model.joint(n).qposadr[0] for n in names],dtype=int)
old=mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'))
old_q=np.load(OUT/'forearm_support_rrt.npz')['qpos'][0]
base=model.qpos0.copy()
for j in range(old.njnt):
    k=model.joint(old.joint(j).name).id
    n=7 if old.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else (4 if old.jnt_type[j]==mujoco.mjtJoint.mjJNT_BALL else 1)
    base[model.jnt_qposadr[k]:model.jnt_qposadr[k]+n]=old_q[old.jnt_qposadr[j]:old.jnt_qposadr[j]+n]
base[model.joint('g1_a/floating_base').qposadr[0]]=-.3
base[model.joint('training_elbow_hinge').qposadr[0]]=.25
for n in names[-2:]:
    j=model.joint(n).id;a=int(model.jnt_qposadr[j]);lo,hi=model.jnt_range[j]
    base[a]=np.clip(base[a]+.5,lo+.03,hi-.03)
goal=base.copy();goal[qa]=candidate[qa]
rng=np.random.default_rng(20261109)
checks=0
def clear(q):
    global checks
    checks+=1
    data.qpos[:]=base;data.qpos[qa]=q
    mujoco.mj_forward(model,data)
    for c in data.contact:
        if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
        names_contact=[model.geom(g).name for g in (c.geom1,c.geom2)]
        bodies=[model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
        if 'floor' not in names_contact and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):return False
    return True
def edge(a,b):
    n=max(2,int(np.ceil(np.linalg.norm(b-a)/.035)))
    return all(clear((1-u)*a+u*b) for u in np.linspace(0,1,n+1)[1:])
def extend(nodes,parents,target,step=.12):
    arr=np.asarray(nodes);near=int(np.argmin(np.sum((arr-target)**2,axis=1)))
    delta=target-nodes[near];norm=np.linalg.norm(delta)
    q=nodes[near]+delta*min(1.,step/max(norm,1e-12))
    if not edge(nodes[near],q):return None
    nodes.append(q);parents.append(near);return len(nodes)-1
def trace(nodes,parents,i):
    result=[]
    while i>=0:result.append(nodes[i]);i=parents[i]
    return result[::-1]
start_ok=clear(base[qa]);goal_ok=clear(goal[qa])
safe_start_search=0
if not start_ok:
    original=base[qa].copy();chosen=None;best=float('inf')
    palm=model.body('g1_a/wuji_left/l_wrist').id
    for _ in range(15000):
        safe_start_search+=1
        ref=original if rng.random()<.5 else goal[qa]
        q=ref+rng.normal(0,rng.uniform(.18,.85),len(qa))
        for i,n in enumerate(names):
            lo,hi=model.jnt_range[model.joint(n).id];q[i]=np.clip(q[i],lo+.04,hi-.04)
        if not clear(q):continue
        wrist=data.xpos[palm]
        if wrist[0]>-.12 or wrist[2]<.7:continue
        score=float(np.linalg.norm(q-original))+.3*float(np.linalg.norm(q-goal[qa]))
        if score<best:best=score;chosen=q.copy()
    if chosen is not None:
        base[qa]=chosen
        start_ok=clear(base[qa])
data.qpos[:]=base;mujoco.mj_forward(model,data)
start_deep=[{'bodies':[model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)],
             'distance_m':float(c.dist)} for c in data.contact
            if c.geom1>=0 and c.geom2>=0 and c.dist<-.003 and
            'floor' not in (model.geom(c.geom1).name,model.geom(c.geom2).name) and
            any(model.body(model.geom_bodyid[g]).name.startswith(('g1_a/','g1_b/')) for g in (c.geom1,c.geom2))]
found=None;iterations=0
if start_ok and goal_ok:
    if edge(base[qa],goal[qa]):found=[base[qa],goal[qa]]
    else:
        trees=[([base[qa].copy()],[-1]),([goal[qa].copy()],[-1])]
        for iteration in range(3000):
            iterations=iteration+1
            sample=goal[qa] if rng.random()<.15 else base[qa]+rng.random()*(goal[qa]-base[qa])+rng.normal(0,.3,len(qa))
            active=iteration%2;other=1-active
            ni=extend(*trees[active],sample)
            if ni is None:continue
            target=trees[active][0][ni]
            for _ in range(25):
                nj=extend(*trees[other],target)
                if nj is None:break
                if np.linalg.norm(trees[other][0][nj]-target)<1e-7:
                    found=(trace(*trees[0],ni)+trace(*trees[1],nj)[::-1] if active==0
                           else trace(*trees[0],nj)+trace(*trees[1],ni)[::-1])
                    break
            if found is not None:break
if found is not None:
    for _ in range(150):
        if len(found)<3:break
        i,j=sorted(rng.choice(len(found),2,replace=False))
        if j>i+1 and edge(found[i],found[j]):found=found[:i+1]+found[j:]
    path=np.repeat(base[None,:],len(found),axis=0)
    for row,q in zip(path,found):row[qa]=q
    np.savez_compressed(OUT/'dynamic_palm_support_rrt.npz',qpos=path)
report={'scope':'Static path at prescribed elbow angle .25 rad; no dynamic execution claim',
        'start_clear':start_ok,'safe_start_search_samples':safe_start_search,'start_deep_contacts':start_deep,'goal_clear':goal_ok,'found_path':found is not None,
        'waypoints':len(found) if found is not None else 0,'collision_checks':checks,
        'iterations':iterations,'dense_recheck_passed':all(edge(a,b) for a,b in zip(found,found[1:])) if found is not None else False}
(OUT/'dynamic_palm_support_rrt_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
