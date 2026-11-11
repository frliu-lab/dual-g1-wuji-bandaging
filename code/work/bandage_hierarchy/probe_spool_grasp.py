"""Static B-left Wuji hand approach to the loose bandage spool in the composite scene."""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--goal-z',type=float,default=.72)
parser.add_argument('--y-closure',type=float,default=0.)
parser.add_argument('--b-root-x',type=float,default=.7)
parser.add_argument('--spool-z',type=float,default=.72,help='World-height initial state for spool free joint')
parser.add_argument('--warm-dynamic',action='store_true',help='Solve from the settled-spool dynamic terminal state')
parser.add_argument('--warm-joint',action='store_true',help='Use terminal state of staged right-palm and elbow-hold joint rollout')
args=parser.parse_args()
if not .70<=args.goal_z<=.77:parser.error('Goal height outside spool vicinity')
if not 0.<=args.y_closure<=.008:parser.error('Y closure must be in [0,0.008] m')
if not .5<=args.b_root_x<=.7:parser.error('B root x must be in [0.5,0.7] m')
if not .69<=args.spool_z<=.72:parser.error('Spool center z must be in [0.69,0.72] m')
if args.warm_joint and not args.warm_dynamic:parser.error('Joint warm state requires --warm-dynamic')
scene=ET.parse(OUT/'joint_support_pinch_variant.xml').getroot()
if args.b_root_x!=.7:
    scene.find(".//body[@name='g1_b/pelvis']").set('pos',f'{args.b_root_x} 0 .8')
model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
data=mujoco.MjData(model)
base=np.load(OUT/'joint_support_pinch_static_candidate.npz')['qpos'].copy()
if args.warm_dynamic:
    if abs(args.b_root_x-.55)>1e-9 or abs(args.spool_z-.702)>1e-9:
        parser.error('Dynamic warm start requires B root x=0.55 and spool z=0.702')
    warm_name=('joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_palmgoal11p5_elbowweight10p0_spool_settled_bwarm_rewrist_roll_final_state.npz'
               if args.warm_joint else 'joint_support_pinch_staged_wbc_report_spool_settled_final_state.npz')
    base=np.load(OUT/warm_name)['qpos'].copy()
else:
    base[model.joint('g1_b/floating_base').qposadr[0]]=args.b_root_x
    base[model.joint('bandage_spool_free').qposadr[0]+2]=args.spool_z
joint_ids=[j for j in range(model.njnt)
           if (model.joint(j).name.startswith('g1_b/left_') and
               any(s in model.joint(j).name for s in ('shoulder','elbow','wrist')))
           or model.joint(j).name.startswith(('g1_b/wuji_left/l_thumb_',
                                               'g1_b/wuji_left/l_index_finger_'))]
qa=np.array([model.jnt_qposadr[j] for j in joint_ids],dtype=int)
va=np.array([model.jnt_dofadr[j] for j in joint_ids],dtype=int)
lo=np.array([model.jnt_range[j,0]+.03 for j in joint_ids])
hi=np.array([model.jnt_range[j,1]-.03 for j in joint_ids])
start=np.clip(base[qa],lo+1e-5,hi-1e-5)
tips=[model.body('g1_b/wuji_left/l_index_finger_tip_sensor_frame').id,
      model.body('g1_b/wuji_left/l_thumb_tip_sensor_frame').id]
if args.warm_dynamic:
    data.qpos[:]=base;mujoco.mj_forward(model,data)
    spool_center=data.xpos[model.body('bandage_spool').id]
    goals=np.array([[spool_center[0],spool_center[1]-.06+args.y_closure,spool_center[2]+args.goal_z-.702],
                    [spool_center[0],spool_center[1]+.06-args.y_closure,spool_center[2]+args.goal_z-.702]])
else:
    goals=np.array([[.38,-.50+args.y_closure,args.goal_z],
                    [.38,-.38-args.y_closure,args.goal_z]])
def forward(x):
    data.qpos[:]=base;data.qpos[qa]=x;mujoco.mj_forward(model,data)
def residual(x):
    forward(x)
    return np.r_[30*(data.xpos[tips]-goals).ravel(),.04*(x-start)]
def jacobian(x):
    forward(x);J=np.zeros((6+len(x),len(x)))
    for i,bid in enumerate(tips):
        jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,data.xpos[bid],bid)
        J[3*i:3*i+3]=30*jp[:,va]
    J[6:]=.04*np.eye(len(x));return J
fit=least_squares(residual,start,jac=jacobian,bounds=(lo,hi),max_nfev=140)
def objective(x):
    forward(x)
    penetration=0.
    for c in data.contact:
        if c.geom1<0 or c.geom2<0:continue
        geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
        bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                model.body(model.geom_bodyid[c.geom2]).name)
        if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
            penetration+=max(0.,-float(c.dist)-.002)**2
    return float(1e4*np.sum((data.xpos[tips]-goals)**2)+2e5*penetration+
                 .002*np.sum((x-start)**2))
refine=minimize(objective,fit.x,method='L-BFGS-B',bounds=list(zip(lo,hi)),
                options={'maxiter':100,'ftol':1e-8})
forward(refine.x)
deep=[];forces={'thumb':0.,'index':0.};normals={'thumb':np.zeros(3),'index':np.zeros(3)}
for ci,c in enumerate(data.contact):
    if c.geom1<0 or c.geom2<0:continue
    geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
    bodies=(model.body(model.geom_bodyid[c.geom1]).name,
            model.body(model.geom_bodyid[c.geom2]).name)
    if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
        deep.append({'bodies':bodies,'geoms':geoms,'distance_m':float(c.dist)})
    if 'bandage_roll' in geoms or 'bandage_core' in geoms:
        for digit in forces:
            if any(b.startswith(f'g1_b/wuji_left/l_{digit}_') for b in bodies):
                wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                force=max(0.,float(wrench[0]))
                forces[digit]+=force
                sign=1. if any(g in ('bandage_roll','bandage_core') for g in geoms[:1]) else -1.
                normals[digit]+=sign*force*np.asarray(c.frame[:3])
a,b=normals.values()
cosine=(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
        if np.linalg.norm(a)*np.linalg.norm(b)>1e-8 else None)
eligible=bool(not deep and all(f>.05 for f in forces.values()) and cosine is not None and cosine<-.5)
report={'scope':'Static B-left spool-contact IK in composite scene; no path, dynamic grasp, or bandaging claim',
        'b_root_x_m':args.b_root_x,
        'spool_initial_z_m':args.spool_z,
        'warm_dynamic':args.warm_dynamic,
        'warm_joint':args.warm_joint,
        'goals_m':goals.tolist(),'joint_names':[model.joint(j).name for j in joint_ids],
        'spool_center_m':data.xpos[model.body('bandage_spool').id].tolist(),
        'tip_positions_m':data.xpos[tips].tolist(),
        'tip_errors_m':np.linalg.norm(data.xpos[tips]-goals,axis=1).tolist(),
        'least_squares_success':bool(fit.success),'collision_refine_success':bool(refine.success),
        'deep_contacts':deep,'forces_n':forces,'normal_cosine':cosine,
        'eligible_static_endpoint':eligible}
stem='joint_spool_grasp_static'+('' if args.goal_z==.72 else '_z'+str(args.goal_z).replace('.','p'))
if args.y_closure:stem+='_closure'+str(args.y_closure).replace('.','p')
if args.b_root_x!=.7:stem+='_root'+str(args.b_root_x).replace('.','p')
if args.spool_z!=.72:stem+='_spoolz'+str(args.spool_z).replace('.','p')
if args.warm_dynamic:stem+='_dynamicwarm'
if args.warm_joint:stem+='_jointwarm'
(OUT/f'{stem}_screen.json').write_text(json.dumps(report,indent=2),encoding='utf8')
if eligible:
    np.savez_compressed(OUT/f'{stem}_candidate.npz',qpos=data.qpos.copy())
    (OUT/f'{stem}_variant.xml').write_text(ET.tostring(scene,encoding='unicode'),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ('joint_names','deep_contacts')}
                 | {'deep_count':len(deep)}))
