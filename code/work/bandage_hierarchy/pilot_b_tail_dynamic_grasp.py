"""Rigid tethered tail proxy grasp pilot, not flexible bandage wrapping."""
import json
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--radius',type=float,default=.009)
parser.add_argument('--tip-servo',action='store_true')
parser.add_argument('--squeeze-mm',type=float,default=1.)
parser.add_argument('--tip-weight',type=float,default=1000.)
parser.add_argument('--normal-goal-n',type=float,default=None)
args=parser.parse_args()
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC

screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
path=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
suffix=f'_{round(args.radius*1000):g}mm' if args.radius!=.009 else ''
seed=json.loads((OUT/f'joint_b_tail_static_proxy_contact_fine_screen{suffix}.json').read_text())['best_candidates'][0]
pos=np.asarray(seed['proxy_pos_m'])
root=ET.parse(OUT/screen['scene']).getroot()
world=root.find('worldbody')
ET.SubElement(world,'geom',{'name':'bandage_tail_rest','type':'cylinder',
                            'pos':f'{pos[0]} {pos[1]} {pos[2]-.024}',
                            'size':'.005 .017','rgba':'.45 .47 .50 1',
                            'friction':'1 .005 .0001'})
tab=ET.SubElement(world,'body',{'name':'bandage_tail_dynamic_proxy',
                                'pos':' '.join(map(str,pos))})
ET.SubElement(tab,'freejoint',{'name':'bandage_tail_proxy_free'})
ET.SubElement(tab,'geom',{'name':'bandage_tail_proxy_contact','type':'cylinder',
                          'quat':'.70710678 0 .70710678 0','size':f'{args.radius} .025',
                          'mass':'.008','rgba':'.95 .89 .70 1','friction':'1 .005 .0001'})
ET.SubElement(tab,'site',{'name':'bandage_tail_proxy_anchor','size':'.002'})
tendon=root.find('tendon')
line=ET.SubElement(tendon,'spatial',{'name':'bandage_tail_proxy_tether','width':'.007',
                                    'rgba':'.95 .89 .70 1','stiffness':'10',
                                    'damping':'.1','springlength':'.683'})
ET.SubElement(line,'site',{'site':'bandage_spool_anchor'})
ET.SubElement(line,'site',{'site':'bandage_tail_proxy_anchor'})
xml_path=OUT/f'joint_b_tail_dynamic_proxy_variant{suffix}.xml'
ET.ElementTree(root).write(xml_path,encoding='utf8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(xml_path))
data=mujoco.MjData(model)
baseq=np.asarray(path['samples'][-1]['qpos'])
assert model.nq==len(baseq)+7
data.qpos[:len(baseq)]=baseq
free_adr=model.joint('bandage_tail_proxy_free').qposadr[0]
data.qpos[free_adr:free_adr+7]=[*pos,1,0,0,0]
for digit,key in (('thumb','thumb_flex_rad'),('index','index_flex_rad')):
    for j in range(model.njnt):
        name=model.joint(j).name
        if name.startswith(f'g1_b/wuji_right/r_{digit}_') and 'abd' not in name:
            data.qpos[model.jnt_qposadr[j]]=np.clip(seed[key],model.jnt_range[j,0]+.01,
                                                      model.jnt_range[j,1]-.01)
mujoco.mj_forward(model,data)
initial_tab=data.xpos[model.body('bandage_tail_dynamic_proxy').id].copy()
palms=('g1_a/wuji_left/l_wrist','g1_a/wuji_right/r_wrist',
       'g1_b/wuji_left/l_wrist','g1_b/wuji_right/r_wrist')
def robot_contact_only(m,c):
    return any(m.body(m.geom_bodyid[int(g)]).name.startswith(('g1_a/','g1_b/'))
               for g in c.geom if g>=0)
wbc=JointWBC(model,data.qpos.copy(),palm_names=palms,contact_filter=robot_contact_only)
wbc.backend='CLARABEL' if args.tip_servo else 'DIRECT'
wbc.tip_position_weight=args.tip_weight
wbc.tail_proxy_digit_normal_goal_n=args.normal_goal_n
tip_bodies={digit:model.body(f'g1_b/wuji_right/r_{digit}_tip_sensor_frame').id
            for digit in ('thumb','index_finger')}
tip_offsets={digit:data.xpos[bid].copy()-initial_tab for digit,bid in tip_bodies.items()}
tip_offsets['thumb'][0]-=args.squeeze_mm/1000
tip_offsets['index_finger'][0]+=args.squeeze_mm/1000
fractions={'thumb':0,'index':0,'opposing':0,'two_feet':0}
max_tab_drift=0.;max_pen=0.;error=None;direct_error=None;steps=0;failure_contacts=[]
duration=.5;interval=round(.02/model.opt.timestep)
for _ in range(round(duration/.02)):
    try:
        if args.tip_servo:
            tab_now=data.xpos[model.body('bandage_tail_dynamic_proxy').id].copy()
            wbc.tip_position_targets={model.body(bid).name:tab_now+tip_offsets[digit]
                                      for digit,bid in tip_bodies.items()}
        ctrl,meta,_=wbc.solve(data)
        data.ctrl[:]=ctrl
        steps+=1
        for _ in range(interval):
            mujoco.mj_step(model,data)
            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError('Nonfinite state')
            max_tab_drift=max(max_tab_drift,float(np.linalg.norm(
                data.xpos[model.body('bandage_tail_dynamic_proxy').id]-initial_tab)))
            normals={'thumb':[],'index':[]};floor=set()
            for c in data.contact:
                names=[model.geom(g).name for g in c.geom]
                bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom]
                if 'floor' in names:
                    for side in ('left','right'):
                        if any(n.startswith(f'g1_b/{side}_official_foot_contact_') for n in names):
                            floor.add(side)
                elif any(n.startswith('g1_b/') for n in bodies):
                    max_pen=max(max_pen,-float(c.dist))
                if 'bandage_tail_dynamic_proxy' not in bodies or c.dist>.001:
                    continue
                for digit in normals:
                    if any(n.startswith(f'g1_b/wuji_right/r_{digit}_') for n in bodies):
                        normal=c.frame[:3]*(1 if names[0]=='bandage_tail_proxy_contact' else -1)
                        normals[digit].append(normal)
            for digit in normals:
                fractions[digit]+=bool(normals[digit])
            fractions['opposing']+=any(float(np.dot(a,b))<-.5 for a in normals['thumb'] for b in normals['index'])
            fractions['two_feet']+=floor=={'left','right'}
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        failure_contacts=sorted([{'geoms':[model.geom(g).name for g in c.geom],
                                  'bodies':[model.body(model.geom_bodyid[g]).name for g in c.geom],
                                  'distance_m':float(c.dist)} for c in data.contact],
                                key=lambda row:row['distance_m'])[:12]
        try:
            from direct_wbc_qp import solve_direct
            solve_direct(wbc.last_numeric)
        except Exception as direct_exc:
            direct_error=f'{type(direct_exc).__name__}: {direct_exc}'
        break
den=max(1,steps*interval)
report={'scope':'Dynamic rigid tail proxy with massless tendon to spool; no flexible cloth or wrapping',
        'seed':seed,'tip_servo':args.tip_servo,'squeeze_mm':args.squeeze_mm,
        'tip_position_weight':args.tip_weight,
        'tail_digit_normal_goal_n':args.normal_goal_n,
        'duration_s':float(data.time),'error':error,
        'direct_solver_error_at_failure':direct_error,'control_solves':steps,
        'failure_contacts':failure_contacts,
        'contact_fractions':{k:v/den for k,v in fractions.items()},
        'max_tail_center_drift_m':max_tab_drift,
        'max_b_nonfloor_penetration_m':max_pen,
        'finite_state':bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))}
(OUT/f'joint_b_tail_dynamic_proxy_grasp_pilot{suffix}{"_tip_servo" if args.tip_servo else ""}{f"_{args.squeeze_mm:g}mm_w{args.tip_weight:g}" if args.tip_servo else ""}{f"_force{args.normal_goal_n:g}n" if args.normal_goal_n is not None else ""}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
