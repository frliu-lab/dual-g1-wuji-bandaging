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
parser.add_argument('--pose-servo',action='store_true')
parser.add_argument('--optimized-tip-seed',action='store_true')
parser.add_argument('--tip-pinch-seed',action='store_true')
parser.add_argument('--tip-pinch-rank',type=int,default=0)
args=parser.parse_args()
if args.optimized_tip_seed and args.tip_pinch_seed:
    parser.error('Choose one alternate seed')
sys.path.insert(0,str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC

screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
path=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
suffix=f'_{round(args.radius*1000):g}mm' if args.radius!=.009 else ''
seed=json.loads((OUT/f'joint_b_tail_static_proxy_contact_fine_screen{suffix}.json').read_text())['best_candidates'][0]
tip_ik=None
tip_pinch=None
if args.optimized_tip_seed:
    tip_ik=json.loads((OUT/'joint_b_tail_arm_digit_ik_screen.json').read_text())
    seed=tip_ik['results'][2]
if args.tip_pinch_seed:
    tip_pinch=json.loads((OUT/'joint_b_tail_tip_pinch_static_screen.json').read_text())
    seed=tip_pinch['best_candidates'][args.tip_pinch_rank]
pos=np.asarray(tip_ik['proxy_center_m'] if tip_ik else
               seed['proxy_center_m'] if tip_pinch else seed['proxy_pos_m'])
geom_quat=np.asarray(tip_ik['proxy_quaternion_wxyz'] if tip_ik else
                     seed['proxy_quaternion_wxyz'] if tip_pinch else
                     [.70710678,0,.70710678,0])
root=ET.parse(OUT/screen['scene']).getroot()
world=root.find('worldbody')
ET.SubElement(world,'geom',{'name':'bandage_tail_rest','type':'cylinder',
                            'pos':f'{pos[0]} {pos[1]} {pos[2]-(.029 if tip_pinch else .028 if tip_ik else .024)}',
                            'size':f'.005 {(.014 if tip_ik or tip_pinch else .017)}','rgba':'.45 .47 .50 1',
                            'friction':'1 .005 .0001'})
tab=ET.SubElement(world,'body',{'name':'bandage_tail_dynamic_proxy',
                                'pos':' '.join(map(str,pos))})
ET.SubElement(tab,'freejoint',{'name':'bandage_tail_proxy_free'})
ET.SubElement(tab,'geom',{'name':'bandage_tail_proxy_contact','type':'cylinder',
                          'quat':' '.join(map(str,geom_quat)),'size':f'{args.radius} .025',
                          'mass':'.008','rgba':'.95 .89 .70 1','friction':'1 .005 .0001'})
ET.SubElement(tab,'site',{'name':'bandage_tail_proxy_anchor','size':'.002'})
tendon=root.find('tendon')
line=ET.SubElement(tendon,'spatial',{'name':'bandage_tail_proxy_tether','width':'.007',
                                    'rgba':'.95 .89 .70 1','stiffness':'10',
                                    'damping':'.1','springlength':str(float(np.linalg.norm(
                                        pos-np.array([.38,-.44,.702]))))})
ET.SubElement(line,'site',{'site':'bandage_spool_anchor'})
ET.SubElement(line,'site',{'site':'bandage_tail_proxy_anchor'})
xml_path=OUT/f'joint_b_tail_dynamic_proxy_variant{suffix}{"_optimized_tip" if tip_ik else f"_tip_pinch_rank{args.tip_pinch_rank}" if tip_pinch else ""}.xml'
ET.ElementTree(root).write(xml_path,encoding='utf8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(xml_path))
data=mujoco.MjData(model)
baseq=np.asarray(path['samples'][-1]['qpos'])
assert model.nq==len(baseq)+7
data.qpos[:len(baseq)]=baseq
if tip_ik:
    for name,value in zip(tip_ik['joint_names'],seed['joint_targets_rad']):
        data.qpos[model.joint(name).qposadr[0]]=value
free_adr=model.joint('bandage_tail_proxy_free').qposadr[0]
data.qpos[free_adr:free_adr+7]=[*pos,1,0,0,0]
if not tip_ik:
    for digit,key in (('thumb','thumb_flex_rad'),('index','index_flex_rad')):
        if tip_pinch:
            key=f'{digit}_flex_rad'
        for j in range(model.njnt):
            name=model.joint(j).name
            if name.startswith(f'g1_b/wuji_right/r_{digit}_') and 'abd' not in name:
                data.qpos[model.jnt_qposadr[j]]=np.clip(seed[key],model.jnt_range[j,0]+.01,
                                                          model.jnt_range[j,1]-.01)
mujoco.mj_forward(model,data)
initial_tail_contacts=[{'bodies':[model.body(model.geom_bodyid[g]).name for g in c.geom],
                        'distance_m':float(c.dist)} for c in data.contact if
                       model.geom('bandage_tail_proxy_contact').id in c.geom]
initial_tab=data.xpos[model.body('bandage_tail_dynamic_proxy').id].copy()
tab_id=model.body('bandage_tail_dynamic_proxy').id
initial_tab_rotation=data.xmat[tab_id].reshape(3,3).copy()
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
tip_local_offsets={digit:initial_tab_rotation.T@offset for digit,offset in tip_offsets.items()}
fractions={'thumb':0,'index':0,'opposing':0,'two_feet':0}
max_tab_drift=0.;max_pen=0.;error=None;direct_error=None;steps=0;failure_contacts=[];bad_acc_bounds=[]
control_trace=[]
duration=.5;interval=round(.02/model.opt.timestep)
for _ in range(round(duration/.02)):
    try:
        if args.tip_servo:
            tab_now=data.xpos[tab_id].copy()
            tab_rotation=data.xmat[tab_id].reshape(3,3)
            wbc.tip_position_targets={model.body(bid).name:tab_now+
                (tab_rotation@tip_local_offsets[digit] if args.pose_servo else tip_offsets[digit])
                for digit,bid in tip_bodies.items()}
        ctrl,meta,_=wbc.solve(data)
        data.ctrl[:]=ctrl
        steps+=1
        interval_contacts={digit:0 for digit in ('thumb','index')}
        interval_opposing=0
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
                interval_contacts[digit]+=bool(normals[digit])
            is_opposing=any(float(np.dot(a,b))<-.5 for a in normals['thumb'] for b in normals['index'])
            fractions['opposing']+=is_opposing
            interval_opposing+=is_opposing
            fractions['two_feet']+=floor=={'left','right'}
        center=data.xpos[tab_id].copy()
        control_trace.append({'time_s':float(data.time),
                              'tail_center_m':center.tolist(),
                              'tail_quaternion':data.xquat[tab_id].tolist(),
                              'thumb_tip_m':data.xpos[tip_bodies['thumb']].tolist(),
                              'index_tip_m':data.xpos[tip_bodies['index_finger']].tolist(),
                              'thumb_contact_fraction':interval_contacts['thumb']/interval,
                              'index_contact_fraction':interval_contacts['index']/interval,
                              'opposing_fraction':interval_opposing/interval})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        numeric=getattr(wbc,'last_numeric',None)
        if numeric is not None:
            for dof in np.flatnonzero(numeric['acc_lo']>numeric['acc_hi']+1e-9):
                j=int(model.dof_jntid[dof])
                bad_acc_bounds.append({'joint':model.joint(j).name,
                    'inversion':float(numeric['acc_lo'][dof]-numeric['acc_hi'][dof]),
                    'q':float(data.qpos[model.jnt_qposadr[j]]),
                    'v':float(data.qvel[dof])})
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
        'seed':seed,'initial_tail_contacts':initial_tail_contacts,
        'optimized_tip_seed':args.optimized_tip_seed,
        'tip_pinch_seed':args.tip_pinch_seed,
        'tip_pinch_rank':args.tip_pinch_rank if tip_pinch else None,
        'tip_servo':args.tip_servo,'squeeze_mm':args.squeeze_mm,
        'tip_position_weight':args.tip_weight,
        'pose_servo':args.pose_servo,
        'tail_digit_normal_goal_n':args.normal_goal_n,
        'duration_s':float(data.time),'error':error,
        'direct_solver_error_at_failure':direct_error,'control_solves':steps,
        'failure_contacts':failure_contacts,
        'bad_acceleration_bounds':bad_acc_bounds,
        'control_trace':control_trace,
        'contact_fractions':{k:v/den for k,v in fractions.items()},
        'max_tail_center_drift_m':max_tab_drift,
        'max_b_nonfloor_penetration_m':max_pen,
        'finite_state':bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))}
(OUT/f'joint_b_tail_dynamic_proxy_grasp_pilot{suffix}{"_optimized_tip" if tip_ik else f"_tip_pinch_rank{args.tip_pinch_rank}" if tip_pinch else ""}{"_tip_servo" if args.tip_servo else ""}{f"_{args.squeeze_mm:g}mm_w{args.tip_weight:g}" if args.tip_servo else ""}{f"_force{args.normal_goal_n:g}n" if args.normal_goal_n is not None else ""}{"_pose" if args.pose_servo else ""}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
