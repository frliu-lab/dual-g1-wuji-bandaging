"""Static RRT-Connect screen for the A-left provisional forearm contact pose."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
seed=np.load(OUT/'forearm_support_root_m0p58_curl0p4_seed.npz')
base=seed['initial_qpos'].copy();goal=seed['qpos'].copy()
scene=ET.parse(OUT/'scene.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos','-.58 0 .8')
m=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
d=mujoco.MjData(m)
qa=np.array([m.jnt_qposadr[j] for j in range(m.njnt)
             if m.joint(j).name.startswith('g1_a/left_')
             and any(t in m.joint(j).name for t in ('shoulder','elbow','wrist'))
             and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE],dtype=int)
q0=base[qa];qg=goal[qa]
rng=np.random.default_rng(20261109)
checks=0
def clear(q):
    global checks
    checks+=1
    d.qpos[:]=base;d.qpos[qa]=q;mujoco.mj_forward(m,d)
    for c in d.contact:
        if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
        names=[m.geom(g).name for g in (c.geom1,c.geom2)]
        bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
        if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):return False
    return True
def edge(a,b):
    n=max(2,int(np.ceil(np.linalg.norm(b-a)/.035)))
    return all(clear((1-u)*a+u*b) for u in np.linspace(0,1,n+1)[1:])
def extend(nodes,parents,target,step=.12):
    arr=np.asarray(nodes)
    near=int(np.argmin(np.sum((arr-target)**2,axis=1)))
    delta=target-nodes[near];dist=np.linalg.norm(delta)
    q=nodes[near]+delta*min(1.,step/max(dist,1e-12))
    if not edge(nodes[near],q):return None
    nodes.append(q);parents.append(near);return len(nodes)-1
def trace(nodes,parents,i):
    out=[]
    while i>=0:out.append(nodes[i]);i=parents[i]
    return out[::-1]
if not clear(q0) or not clear(qg):raise RuntimeError('Endpoint is in deep contact')
trees=[([q0.copy()],[-1]),([qg.copy()],[-1])]
found=None
for iteration in range(2000):
    sample=qg if rng.random()<.15 else q0+rng.random()*(qg-q0)+rng.normal(0,.3,len(q0))
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
    raw=len(found)
    for _ in range(150):
        if len(found)<3:break
        i,j=sorted(rng.choice(len(found),2,replace=False))
        if j>i+1 and edge(found[i],found[j]):found=found[:i+1]+found[j:]
    recheck=all(edge(a,b) for a,b in zip(found,found[1:]))
    full=np.repeat(base[None,:],len(found),axis=0)
    for row,q in zip(full,found):row[qa]=q
    np.savez_compressed(OUT/'forearm_support_rrt.npz',qpos=full)
else:raw=0;recheck=False
report={'scope':'Static A-left support-arm RRT screen; no dynamic contact force validation',
        'random_seed':20261109,'collision_checks':checks,
        'found_path':found is not None,'path_waypoints':len(found) if found is not None else 0,
        'raw_waypoints':raw,'dense_recheck_passed':recheck,
        'iterations':iteration+1,'tree_nodes':[len(t[0]) for t in trees]}
(OUT/'forearm_support_rrt_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
