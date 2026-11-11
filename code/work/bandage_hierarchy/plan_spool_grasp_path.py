"""Static collision-screened B-left approach to a spool grip at x=0.55 m."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--settled',action='store_true',help='Plan against spool resting at z=0.702 m')
args=parser.parse_args()
stem=('joint_spool_grasp_static_z0p73_root0p55_spoolz0p702' if args.settled
      else 'joint_spool_grasp_static_z0p745_root0p55')
model=mujoco.MjModel.from_xml_path(str(OUT/f'{stem}_variant.xml'))
data=mujoco.MjData(model)
goal=np.load(OUT/f'{stem}_candidate.npz')['qpos']
base=np.load(OUT/'joint_support_pinch_static_candidate.npz')['qpos'].copy()
base[model.joint('g1_b/floating_base').qposadr[0]]=.55
if args.settled:base[model.joint('bandage_spool_free').qposadr[0]+2]=.702
names=json.loads((OUT/f'{stem}_screen.json').read_text(encoding='utf8'))['joint_names']
js=[model.joint(name).id for name in names]
qa=np.array([model.jnt_qposadr[j] for j in js],dtype=int)
lo=np.array([model.jnt_range[j,0]+.03 for j in js])
hi=np.array([model.jnt_range[j,1]-.03 for j in js])
start=base[qa];end=goal[qa]
rng=np.random.default_rng(20261110)
checks=0
def clear(q):
    global checks
    checks+=1
    data.qpos[:]=base;data.qpos[qa]=q;mujoco.mj_forward(model,data)
    for c in data.contact:
        if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
        geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
        bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                model.body(model.geom_bodyid[c.geom2]).name)
        if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):return False
    return True
def edge(a,b):
    n=max(2,int(np.ceil(np.linalg.norm(b-a)/.04)))
    return all(clear((1-u)*a+u*b) for u in np.linspace(0,1,n+1)[1:])
def extend(nodes,parents,target,step=.18):
    near=int(np.argmin(np.sum((np.asarray(nodes)-target)**2,axis=1)))
    delta=target-nodes[near];dist=float(np.linalg.norm(delta))
    q=nodes[near]+delta*min(1.,step/max(dist,1e-12))
    if not edge(nodes[near],q):return None
    nodes.append(q);parents.append(near);return len(nodes)-1
def trace(nodes,parents,index):
    out=[]
    while index>=0:out.append(nodes[index]);index=parents[index]
    return out[::-1]
start_ok=clear(start);goal_ok=clear(end);found=None;iterations=0
if start_ok and goal_ok:
    trees=[([start.copy()],[-1]),([end.copy()],[-1])]
    for iteration in range(1800):
        iterations=iteration+1
        sample=end if rng.random()<.2 else np.clip(start+rng.random()*(end-start)+rng.normal(0,.3,len(start)),lo,hi)
        active=iteration%2;other=1-active
        ni=extend(*trees[active],sample)
        if ni is None:continue
        target=trees[active][0][ni]
        for _ in range(30):
            nj=extend(*trees[other],target)
            if nj is None:break
            if np.linalg.norm(trees[other][0][nj]-target)<1e-7:
                found=(trace(*trees[0],ni)+trace(*trees[1],nj)[::-1] if active==0
                       else trace(*trees[0],nj)+trace(*trees[1],ni)[::-1])
                break
        if found is not None:break
if found is not None:
    raw=len(found)
    for _ in range(120):
        if len(found)<3:break
        i,j=sorted(rng.choice(len(found),2,replace=False))
        if j>i+1 and edge(found[i],found[j]):found=found[:i+1]+found[j:]
    dense=all(edge(a,b) for a,b in zip(found,found[1:]))
    path=np.repeat(base[None,:],len(found),axis=0)
    for row,q in zip(path,found):row[qa]=q
    np.savez_compressed(OUT/('joint_spool_grasp_rrt_settled.npz' if args.settled else 'joint_spool_grasp_rrt.npz'),qpos=path)
else:raw=0;dense=False
report={'scope':'Static B-left spool approach at fixed B-root x=0.55 m; no locomotion or dynamic grasp claim',
        'spool_initial_z_m':.702 if args.settled else .72,
        'start_valid':start_ok,'goal_valid':goal_ok,'found_path':found is not None,
        'waypoints':len(found) if found is not None else 0,'raw_waypoints':raw,
        'dense_recheck_passed':dense,'iterations':iterations,'collision_checks':checks}
(OUT/('joint_spool_grasp_rrt_settled_report.json' if args.settled else 'joint_spool_grasp_rrt_report.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
