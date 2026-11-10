"""Simulation-only two-G1 WBC test of a provisional A-left forearm contact path."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC
scene=ET.parse(OUT/'scene.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos','-.58 0 .8')
m=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
d=mujoco.MjData(m)
path=np.load(OUT/'forearm_support_rrt.npz')['qpos']
d.qpos[:]=path[0]
mujoco.mj_forward(m,d)
def contact_filter(model,c):
    return any(model.body(model.geom_bodyid[int(g)]).name.startswith(('g1_a/','g1_b/'))
               for g in c.geom if g>=0)
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist',
       'g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
wbc=JointWBC(m,d.qpos.copy(),palm_names=palms,contact_filter=contact_filter)
root_ids=[m.body(f'{r}/pelvis').id for r in ('g1_a','g1_b')]
root_start=d.xpos[root_ids].copy()
hz=50;segment_s=.6;hold_s=2.;interval=round((1/hz)/m.opt.timestep)
schedule=[(segment,k) for segment in range(1,len(path)) for k in range(round(segment_s*hz))]
schedule += [(len(path)-1,round(segment_s*hz))]*round(hold_s*hz)
support_steps=0;contact_steps=0;deep_count=0;max_penetration=0.;max_force=0.
max_pelvis_shift=0.;steps=0;error=None;forearm_trace=[]
final_half_steps=0;final_half_contact_steps=0
contact_force_by_body={}
last_half_force_by_body={}
for segment,k in schedule:
    u=(k+1)/round(segment_s*hz)
    ease=u*u*(3-2*u)
    wbc.qref[:]=(1-ease)*path[segment-1]+ease*path[segment]
    try:
        ctrl,_,_=wbc.solve(d)
        d.ctrl[:]=ctrl
        for _ in range(interval):
            mujoco.mj_step(m,d);steps+=1
            supported=set();force=0.
            for ci,c in enumerate(d.contact):
                if c.geom1<0 or c.geom2<0:continue
                names=[m.geom(g).name for g in (c.geom1,c.geom2)]
                bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
                if 'floor' in names:
                    for b in bodies:
                        if b.startswith('g1_a/'):supported.add('g1_a')
                        if b.startswith('g1_b/'):supported.add('g1_b')
                elif any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                    if c.dist<-.003:
                        deep_count+=1;max_penetration=max(max_penetration,-float(c.dist))
                    if 'manikin_forearm' in names and any(b.startswith('g1_a/wuji_left/') for b in bodies):
                        wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                        normal=max(0.,float(wrench[0]));force+=normal
                        hand_body=next(b for b in bodies if b.startswith('g1_a/wuji_left/'))
                        contact_force_by_body[hand_body]=contact_force_by_body.get(hand_body,0.)+normal*m.opt.timestep
                        if d.time >= (len(path)-1)*segment_s+hold_s-.5:
                            last_half_force_by_body[hand_body]=last_half_force_by_body.get(hand_body,0.)+normal*m.opt.timestep
            support_steps+=int(supported=={'g1_a','g1_b'})
            contact_steps+=int(force>=.05)
            if d.time >= (len(path)-1)*segment_s+hold_s-.5:
                final_half_steps+=1;final_half_contact_steps+=int(force>=.05)
            max_force=max(max_force,force)
            max_pelvis_shift=max(max_pelvis_shift,float(np.max(np.linalg.norm(d.xpos[root_ids]-root_start,axis=1))))
        forearm_trace.append({'time_s':float(d.time),'normal_force_n':force,
                              'contact':force>=.05})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        break
report={'scope':'Short closed-loop A-left training-manikin forearm support test; no bandaging or clinical safety claim',
        'a_root_x_m':-.58,'hand_curl_rad':.4,'wbc_hz':hz,
        'target_duration_s':(len(path)-1)*segment_s+hold_s,
        'simulated_time_s':float(d.time),'foot_support_fraction':support_steps/max(steps,1),
        'forearm_contact_fraction':contact_steps/max(steps,1),
        'final_half_second_forearm_contact_fraction':final_half_contact_steps/max(final_half_steps,1),
        'peak_forearm_normal_force_n':max_force,
        'forearm_normal_impulse_by_hand_body_ns':contact_force_by_body,
        'final_half_second_forearm_normal_impulse_by_hand_body_ns':last_half_force_by_body,
        'deep_nonfloor_robot_contact_count':deep_count,
        'max_nonfloor_robot_penetration_m':max_penetration,
        'max_pelvis_shift_m':max_pelvis_shift,
        'finite_state':bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))),
        'error':error}
(OUT/'forearm_support_wbc_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
(OUT/'forearm_support_wbc_trace.json').write_text(json.dumps(forearm_trace,indent=2),encoding='utf8')
print(json.dumps(report))
