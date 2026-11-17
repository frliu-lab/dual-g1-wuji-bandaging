"""Short dynamic support pilot restarted from the verified joint grasp endpoint.

The restart drops the original routing history, so only the full rollout can pass.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
sys.path.insert(0,str(ROOT/'work/bandage_hierarchy'))
from joint_wbc import JointWBC
from support_quality import SupportQualityMonitor,SupportSample

p=argparse.ArgumentParser()
p.add_argument('--wrist-offset-rad',type=float,nargs=3,default=(0.,0.,0.))
p.add_argument('--palm-position-weight',type=float,default=1000.)
p.add_argument('--palm-rotation-weight',type=float,default=10.)
p.add_argument('--support-palm-force-goal-n',type=float,default=11.5)
p.add_argument('--duration-s',type=float,default=2.)
args=p.parse_args()
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
state=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel']
mujoco.mj_forward(model,data)
base=state['reference_qpos'].copy()
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist','g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def contact_filter(m,c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/','g1_b/')) for g in c.geom if g>=0)
wbc=JointWBC(model,base,palm_names=palms,contact_filter=contact_filter,
             palm_rotation_weights=[args.palm_rotation_weight,10.,10.,10.],
             palm_position_weights=[args.palm_position_weight,1500.,1000.,1000.])
wbc.forearm_contact_force_limit_n=12.
wbc.forearm_finger_force_limit_n=2.
wbc.support_palm_force_goal_n=args.support_palm_force_goal_n
wbc.b_spool_contact_force_cap_n=1.5
wrist=np.array([model.joint(n).qposadr[0] for n in
                ('g1_a/left_wrist_roll_joint','g1_a/left_wrist_pitch_joint','g1_a/left_wrist_yaw_joint')])
elbow=model.joint('training_elbow_hinge')
monitor=SupportQualityMonitor()
rows=[];errors=[]
interval=round(.02/model.opt.timestep)
for k in range(round(args.duration_s*50)):
    wbc.qref[:]=base
    wbc.qref[wrist]+=min(1.,k*.02/.5)*np.asarray(args.wrist_offset_rad)
    wbc.qref[elbow.qposadr[0]]=.25
    try:
        data.ctrl[:],_,_=wbc.solve(data)
        impulses={};peak=0.;feet=set();deep=False
        for _ in range(interval):
            mujoco.mj_step(model,data)
            frame_force=0.
            for ci,c in enumerate(data.contact):
                if c.geom1<0 or c.geom2<0:continue
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                if 'floor' in geoms:
                    for robot in ('g1_a','g1_b'):
                        if any(b.startswith(robot+'/') for b in bodies):feet.add(robot)
                elif any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:deep=True
                if 'manikin_forearm' in geoms:
                    hand=next((b for b in bodies if b.startswith('g1_a/wuji_left/')),None)
                    if hand:
                        force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
                        normal=max(0.,float(force[0]))
                        impulses[hand]=impulses.get(hand,0.)+normal*model.opt.timestep
                        frame_force+=normal
            peak=max(peak,frame_force)
        valid=monitor.update(SupportSample(float(data.time),float(data.qpos[elbow.qposadr[0]]),
                float(data.qvel[elbow.dofadr[0]]),peak,impulses,feet=={'g1_a','g1_b'},deep))
        rows.append({'time_s':float(data.time),'valid':valid,'elbow_rad':float(data.qpos[elbow.qposadr[0]]),
                     'impulse_by_body_ns':impulses,'window':monitor.last_report})
    except RuntimeError as exc:
        errors.append(str(exc));break
report={'scope':'Checkpoint restart pilot; full-sequence validation required',
        'wrist_offset_rad':args.wrist_offset_rad,'palm_position_weight':args.palm_position_weight,
        'palm_rotation_weight':args.palm_rotation_weight,'support_palm_force_goal_n':args.support_palm_force_goal_n,
        'duration_s':args.duration_s,'completed':len(rows)==round(args.duration_s*50) and not errors,
        'final_quality':monitor.last_report,'final_elbow_rad':rows[-1]['elbow_rad'] if rows else None,
        'errors':errors,'trace':rows}
stem='joint_support_hold_checkpoint_'+str(abs(hash((tuple(args.wrist_offset_rad),args.palm_position_weight,
                                                   args.palm_rotation_weight,args.support_palm_force_goal_n)))%10**10)
(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
