"""Reproducible geometric RRT-Connect screen for the isolated side-pinch variant."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(OUT/'pad_side_pinch_variant.xml'))
d=mujoco.MjData(m)
seeds=np.load(OUT/'pad_side_pinch_candidates.npz')
base=seeds['initial_qpos'].copy()
qa=seeds['joint_qpos_addresses'].astype(int)
q0=base[qa].copy()
joint_by_qadr={int(m.jnt_qposadr[j]):j for j in range(m.njnt)}
lo=np.array([m.jnt_range[joint_by_qadr[i],0]+.03 for i in qa])
hi=np.array([m.jnt_range[joint_by_qadr[i],1]-.03 for i in qa])
rng=np.random.default_rng(20261109)
checks=0
def clear(q):
    global checks
    checks+=1
    d.qpos[:]=base
    d.qpos[qa]=q
    mujoco.mj_forward(m,d)
    for c in d.contact:
        if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
        names=[m.geom(g).name for g in (c.geom1,c.geom2)]
        bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
        if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
            return False
    return True
def edge(a,b):
    n=max(2,int(np.ceil(np.linalg.norm(b-a)/.04)))
    return all(clear((1-u)*a+u*b) for u in np.linspace(0,1,n+1)[1:])
def extend(nodes,parents,target,step=.18):
    arr=np.asarray(nodes)
    near=int(np.argmin(np.sum((arr-target)**2,axis=1)))
    delta=target-nodes[near]
    dist=float(np.linalg.norm(delta))
    candidate=nodes[near]+delta*min(1.,step/max(dist,1e-12))
    if not edge(nodes[near],candidate):return None
    nodes.append(candidate);parents.append(near)
    return len(nodes)-1
def trace(nodes,parents,index):
    out=[]
    while index>=0:
        out.append(nodes[index]);index=parents[index]
    return out[::-1]
reports=[]
for candidate_index,goal_full in enumerate(seeds['qpos']):
    goal=goal_full[qa].copy()
    if not clear(q0) or not clear(goal):
        reports.append({'candidate_index':candidate_index,'valid_endpoints':False})
        continue
    trees=[([q0.copy()],[-1]),([goal.copy()],[-1])]
    found=None
    for iteration in range(1000):
        if rng.random()<.15:
            sample=trees[1][0][0].copy()
        else:
            u=rng.random()
            sample=np.clip((1-u)*q0+u*goal+rng.normal(0,.3,len(q0)),lo,hi)
        active=iteration%2
        other=1-active
        ni=extend(*trees[active],sample)
        if ni is None:continue
        target=trees[active][0][ni]
        for _ in range(30):
            nj=extend(*trees[other],target)
            if nj is None:break
            if np.linalg.norm(trees[other][0][nj]-target)<1e-7:
                if active==0:
                    found=trace(*trees[0],ni)+trace(*trees[1],nj)[::-1]
                else:
                    found=trace(*trees[0],nj)+trace(*trees[1],ni)[::-1]
                break
        if found is not None:break
    if found is not None:
        raw_waypoints=len(found)
        for _ in range(120):
            if len(found)<3:break
            i,j=sorted(rng.choice(len(found),size=2,replace=False))
            if j<=i+1:continue
            if edge(found[i],found[j]):found=found[:i+1]+found[j:]
        dense_safe=all(edge(a,b) for a,b in zip(found,found[1:]))
        path_length=float(sum(np.linalg.norm(b-a) for a,b in zip(found,found[1:])))
        full=np.repeat(base[None,:],len(found),axis=0)
        for row,q in zip(full,found):row[qa]=q
        np.savez_compressed(OUT/f'pad_side_pinch_rrt_{candidate_index}.npz',qpos=full)
    reports.append({'candidate_index':candidate_index,
                    'source_candidate_index':int(seeds['source_indices'][candidate_index]),
                    'valid_endpoints':True,'found_path':found is not None,
                    'path_waypoints':len(found) if found is not None else 0,
                    'raw_waypoints':raw_waypoints if found is not None else 0,
                    'dense_recheck_passed':dense_safe if found is not None else False,
                    'path_joint_space_length_rad':path_length if found is not None else None,
                    'iterations':iteration+1,
                    'tree_nodes':[len(t[0]) for t in trees]})
report={'scope':'Static joint-space RRT-Connect screen; collision threshold 3 mm, no dynamics or grasp proof',
        'random_seed':20261109,'collision_checks':checks,'results':reports}
(OUT/'pad_side_pinch_rrt_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
