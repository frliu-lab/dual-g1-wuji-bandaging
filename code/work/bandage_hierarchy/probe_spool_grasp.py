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
parser.add_argument('--grasp-axis',choices=('x','y'),default='y',help='Opposed fingertip target axis around the upright spool')
parser.add_argument('--partner-digit',choices=('index','middle'),default='index',help='Finger opposed to the thumb')
parser.add_argument('--y-closure',type=float,default=0.)
parser.add_argument('--b-root-x',type=float,default=.7)
parser.add_argument('--b-root-y',type=float,default=0.)
parser.add_argument('--b-root-yaw-deg',type=float,default=0.,help='Prepositioned B-base heading in degrees')
parser.add_argument('--spool-z',type=float,default=.72,help='World-height initial state for spool free joint')
parser.add_argument('--raised-spool',action='store_true',help='Prototype 60-mm tray pedestal with spool center at z=0.76 m')
parser.add_argument('--spool-handle',action='store_true',help='Prototype a grip handle rigidly fixed to the spool core')
parser.add_argument('--handle-radius',type=float,default=.012,help='Prototype handle radius in meters')
parser.add_argument('--handle-clearance',type=float,default=.010,help='Target sensor-frame offset outside handle radius')
parser.add_argument('--handle-solref-time',type=float,default=0.,help='Optional contact time constant for compliant prototype handle')
parser.add_argument('--warm-dynamic',action='store_true',help='Solve from the settled-spool dynamic terminal state')
parser.add_argument('--warm-joint',action='store_true',help='Use terminal state of staged right-palm and elbow-hold joint rollout')
parser.add_argument('--seed-index',type=int,default=0,help='Deterministic alternate IK initialization')
parser.add_argument('--avoid-index-cap',action='store_true',help='Penalize B index distal/middle contact on spool endcap')
parser.add_argument('--low-force-refine',action='store_true',help='Include MuJoCo contact-force and opposed-normal terms in static refinement')
parser.add_argument('--seed-scale',type=float,default=.18,help='Alternate IK seed joint perturbation in radians')
args=parser.parse_args()
if not .70<=args.goal_z<=.80:parser.error('Goal height outside spool vicinity')
if not 0.<=args.y_closure<=.008:parser.error('Y closure must be in [0,0.008] m')
if not .5<=args.b_root_x<=.7:parser.error('B root x must be in [0.5,0.7] m')
if not -.25<=args.b_root_y<=.15:parser.error('B root y must be in [-0.25,0.15] m')
if not -45.<=args.b_root_yaw_deg<=45.:parser.error('B heading must be in [-45,45] degrees')
if not .69<=args.spool_z<=.80:parser.error('Spool center z must be in [0.69,0.80] m')
if args.raised_spool and abs(args.spool_z-.76)>1e-9:parser.error('Raised prototype requires --spool-z 0.76')
if args.handle_solref_time and not (args.spool_handle and .01<=args.handle_solref_time<=.08):
    parser.error('Handle solref time requires --spool-handle and must be in [0.01,0.08] s')
if not .012<=args.handle_radius<=.025 or (args.handle_radius!=.012 and not args.spool_handle):
    parser.error('Handle radius must be in [0.012,0.025] m and requires --spool-handle')
if not 0.<=args.handle_clearance<=.015 or (args.handle_clearance!=.010 and not args.spool_handle):
    parser.error('Handle clearance must be in [0,0.015] m and requires --spool-handle')
if args.warm_joint and not args.warm_dynamic:parser.error('Joint warm state requires --warm-dynamic')
if not 0<=args.seed_index<=100:parser.error('Seed index must be in [0,100]')
if not .05<=args.seed_scale<=.6:parser.error('Seed scale must be in [0.05,0.6] radians')
scene=ET.parse(OUT/'joint_support_pinch_variant.xml').getroot()
if args.raised_spool:
    ET.SubElement(scene.find('worldbody'),'geom',{
        'name':'spool_pedestal_prototype','type':'box','pos':'.38 -.44 .705',
        'size':'.035 .035 .030','rgba':'.52 .56 .61 1',
        'friction':'1 .005 .0001'})
if args.spool_handle:
    handle_attrs={
        'name':'bandage_handle_prototype','type':'cylinder','pos':'0 0 .065',
        'size':f'{args.handle_radius} .040','mass':'.03','rgba':'.45 .47 .50 1',
        'friction':'1 .005 .0001'}
    if args.handle_solref_time:handle_attrs['solref']=f'{args.handle_solref_time} 1'
    ET.SubElement(scene.find(".//body[@name='bandage_spool']"),'geom',handle_attrs)
if args.b_root_x!=.7 or args.b_root_y!=0.:
    scene.find(".//body[@name='g1_b/pelvis']").set('pos',f'{args.b_root_x} {args.b_root_y} .8')
model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
data=mujoco.MjData(model)
base=np.load(OUT/'joint_support_pinch_static_candidate.npz')['qpos'].copy()
if args.warm_dynamic:
    if abs(args.b_root_x-.55)>1e-9 or abs(args.b_root_y)>1e-9 or abs(args.spool_z-.702)>1e-9:
        parser.error('Dynamic warm start requires B root x=0.55, y=0, and spool z=0.702')
    warm_name=('joint_support_pinch_staged_wbc_report_position1500p0_stageright_specialist_palmgoal11p5_elbowweight10p0_spool_settled_bwarm_rewrist_roll_final_state.npz'
               if args.warm_joint else 'joint_support_pinch_staged_wbc_report_spool_settled_final_state.npz')
    base=np.load(OUT/warm_name)['qpos'].copy()
else:
    base[model.joint('g1_b/floating_base').qposadr[0]]=args.b_root_x
    base[model.joint('g1_b/floating_base').qposadr[0]+1]=args.b_root_y
    if args.b_root_yaw_deg:
        yaw=(np.pi+np.deg2rad(args.b_root_yaw_deg))/2
        adr=model.joint('g1_b/floating_base').qposadr[0]
        base[adr+3:adr+7]=[np.cos(yaw),0.,0.,np.sin(yaw)]
    base[model.joint('bandage_spool_free').qposadr[0]+2]=args.spool_z
joint_ids=[j for j in range(model.njnt)
           if (model.joint(j).name.startswith('g1_b/left_') and
               any(s in model.joint(j).name for s in ('shoulder','elbow','wrist')))
           or model.joint(j).name.startswith(('g1_b/wuji_left/l_thumb_',
                                               f'g1_b/wuji_left/l_{args.partner_digit}_finger_'))]
qa=np.array([model.jnt_qposadr[j] for j in joint_ids],dtype=int)
va=np.array([model.jnt_dofadr[j] for j in joint_ids],dtype=int)
lo=np.array([model.jnt_range[j,0]+.03 for j in joint_ids])
hi=np.array([model.jnt_range[j,1]-.03 for j in joint_ids])
start=np.clip(base[qa],lo+1e-5,hi-1e-5)
if args.seed_index:
    rng=np.random.default_rng(20261111+args.seed_index)
    start=np.clip(start+rng.normal(0,args.seed_scale,len(start)),lo+1e-5,hi-1e-5)
tips=[model.body(f'g1_b/wuji_left/l_{args.partner_digit}_finger_tip_sensor_frame').id,
      model.body('g1_b/wuji_left/l_thumb_tip_sensor_frame').id]
if args.warm_dynamic:
    data.qpos[:]=base;mujoco.mj_forward(model,data)
    spool_center=data.xpos[model.body('bandage_spool').id]
    goals=np.array([[spool_center[0],spool_center[1]-.06+args.y_closure,spool_center[2]+args.goal_z-.702],
                    [spool_center[0],spool_center[1]+.06-args.y_closure,spool_center[2]+args.goal_z-.702]])
else:
    if args.spool_handle and args.grasp_axis=='x':
        gap=args.handle_radius+args.handle_clearance
        goals=np.array([[.38-gap+args.y_closure,-.44,args.goal_z],
                        [.38+gap-args.y_closure,-.44,args.goal_z]])
    elif args.spool_handle:
        gap=args.handle_radius+args.handle_clearance
        goals=np.array([[.38,-.44-gap+args.y_closure,args.goal_z],
                        [.38,-.44+gap-args.y_closure,args.goal_z]])
    elif args.grasp_axis=='x':
        goals=np.array([[.38-.06+args.y_closure,-.44,args.goal_z],
                        [.38+.06-args.y_closure,-.44,args.goal_z]])
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
    cap_penalty=0.
    contact_forces={'thumb':0.,args.partner_digit:0.}
    contact_normals={'thumb':np.zeros(3),args.partner_digit:np.zeros(3)}
    for ci,c in enumerate(data.contact):
        if c.geom1<0 or c.geom2<0:continue
        geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
        bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                model.body(model.geom_bodyid[c.geom2]).name)
        if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
            penetration+=max(0.,-float(c.dist)-.002)**2
        if (args.avoid_index_cap and 'bandage_roll' in geoms and
            any(b.startswith(f'g1_b/wuji_left/l_{args.partner_digit}_finger_') and
                any(s in b for s in ('distal','middle')) for b in bodies) and
            abs(float(c.frame[2]))>.7):
            cap_penalty+=max(0.,.002-float(c.dist))**2
        if args.low_force_refine and any(g in geoms for g in ('bandage_roll','bandage_core','bandage_handle_prototype')):
            for digit in contact_forces:
                if any(b.startswith(f'g1_b/wuji_left/l_{digit}_') for b in bodies):
                    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                    force=max(0.,float(wrench[0]))
                    contact_forces[digit]+=force
                    sign=1. if geoms[0] in ('bandage_roll','bandage_core','bandage_handle_prototype') else -1.
                    contact_normals[digit]+=sign*force*np.asarray(c.frame[:3])
    force_penalty=0.
    if args.low_force_refine:
        force_penalty=sum(2e2*max(0.,f-1.5)**2 + 5e2*max(0.,.15-f)**2
                          for f in contact_forces.values())
        a,b=contact_normals.values()
        denom=np.linalg.norm(a)*np.linalg.norm(b)
        if denom>1e-8:
            force_penalty+=3e2*max(0.,float(a@b/denom)+.6)**2
    return float(1e4*np.sum((data.xpos[tips]-goals)**2)+2e5*penetration+2e6*cap_penalty+force_penalty+
                 .002*np.sum((x-start)**2))
refine=minimize(objective,fit.x,method='L-BFGS-B',bounds=list(zip(lo,hi)),
                options={'maxiter':100,'ftol':1e-8})
forward(refine.x)
deep=[];cap_contacts=[];spool_contacts=[];forces={'thumb':0.,args.partner_digit:0.};normals={'thumb':np.zeros(3),args.partner_digit:np.zeros(3)}
for ci,c in enumerate(data.contact):
    if c.geom1<0 or c.geom2<0:continue
    geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
    bodies=(model.body(model.geom_bodyid[c.geom1]).name,
            model.body(model.geom_bodyid[c.geom2]).name)
    if 'floor' not in geoms and any(b.startswith(('g1_a/','g1_b/')) for b in bodies) and c.dist<-.003:
        deep.append({'bodies':bodies,'geoms':geoms,'distance_m':float(c.dist)})
    if any(g in geoms for g in ('bandage_roll','bandage_core','bandage_handle_prototype')):
        if ('bandage_roll' in geoms and
            any(b.startswith(f'g1_b/wuji_left/l_{args.partner_digit}_finger_') and
                any(s in b for s in ('distal','middle')) for b in bodies) and
            abs(float(c.frame[2]))>.7):
            cap_contacts.append({'body':next(b for b in bodies if b.startswith('g1_b/wuji_left/')),
                                 'distance_m':float(c.dist),'normal':np.asarray(c.frame[:3]).tolist()})
        for digit in forces:
            if any(b.startswith(f'g1_b/wuji_left/l_{digit}_') for b in bodies):
                wrench=np.zeros(6);mujoco.mj_contactForce(model,data,ci,wrench)
                force=max(0.,float(wrench[0]))
                forces[digit]+=force
                sign=1. if geoms[0] in ('bandage_roll','bandage_core','bandage_handle_prototype') else -1.
                normals[digit]+=sign*force*np.asarray(c.frame[:3])
                spool_contacts.append({'digit':digit,'bodies':bodies,'geoms':geoms,
                                       'distance_m':float(c.dist),'force_n':force,
                                       'spool_to_finger_normal':(sign*np.asarray(c.frame[:3])).tolist()})
a,b=normals.values()
cosine=(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
        if np.linalg.norm(a)*np.linalg.norm(b)>1e-8 else None)
eligible=bool(not deep and (not args.avoid_index_cap or not cap_contacts) and
              all(f>.05 for f in forces.values()) and cosine is not None and cosine<-.5)
tip_errors=np.linalg.norm(data.xpos[tips]-goals,axis=1)
strict_eligible=bool(eligible and np.max(tip_errors)<=.01 and
                     all(.05<f<=2. for f in forces.values()) and not cap_contacts)
report={'scope':'Static B-left spool-contact IK in composite scene; no path, dynamic grasp, or bandaging claim',
        'b_root_x_m':args.b_root_x,
        'b_root_y_m':args.b_root_y,
        'b_root_yaw_deg':args.b_root_yaw_deg,
        'grasp_axis':args.grasp_axis,
        'partner_digit':args.partner_digit,
        'spool_initial_z_m':args.spool_z,
        'raised_spool_prototype':args.raised_spool,
        'spool_handle_prototype':args.spool_handle,
        'handle_radius_m':args.handle_radius if args.spool_handle else None,
        'handle_clearance_m':args.handle_clearance if args.spool_handle else None,
        'handle_solref_time_s':args.handle_solref_time,
        'warm_dynamic':args.warm_dynamic,
        'warm_joint':args.warm_joint,
        'seed_index':args.seed_index,
        'seed_scale_rad':args.seed_scale,
        'avoid_index_cap':args.avoid_index_cap,
        'low_force_refine':args.low_force_refine,
        'goals_m':goals.tolist(),'joint_names':[model.joint(j).name for j in joint_ids],
        'spool_center_m':data.xpos[model.body('bandage_spool').id].tolist(),
        'tip_positions_m':data.xpos[tips].tolist(),
        'tip_errors_m':tip_errors.tolist(),
        'least_squares_success':bool(fit.success),'collision_refine_success':bool(refine.success),
        'deep_contacts':deep,'index_cap_contacts':cap_contacts,
        'partner_cap_contacts':cap_contacts,'spool_contacts':spool_contacts,
        'forces_n':forces,'normal_cosine':cosine,
        'eligible_static_endpoint':eligible,
        'strict_static_endpoint':strict_eligible,
        'strict_screen_limits':{'max_tip_goal_error_m':.01,'max_normal_force_per_digit_n':2.,
                                'no_index_endcap_contact':True}}
stem='joint_spool_grasp_static'+('' if args.goal_z==.72 else '_z'+str(args.goal_z).replace('.','p'))
if args.grasp_axis!='y':stem+='_axis'+args.grasp_axis
if args.partner_digit!='index':stem+='_partner_'+args.partner_digit
if args.y_closure:stem+='_closure'+str(args.y_closure).replace('.','p')
if args.b_root_x!=.7:stem+='_root'+str(args.b_root_x).replace('.','p')
if args.b_root_y:stem+='_rooty'+str(args.b_root_y).replace('-','m').replace('.','p')
if args.b_root_yaw_deg:stem+='_rootyaw'+str(args.b_root_yaw_deg).replace('-','m').replace('.','p')
if args.spool_z!=.72:stem+='_spoolz'+str(args.spool_z).replace('.','p')
if args.warm_dynamic:stem+='_dynamicwarm'
if args.warm_joint:stem+='_jointwarm'
if args.seed_index:stem+='_seed'+str(args.seed_index)
if args.seed_index and args.seed_scale!=.18:stem+='_scale'+str(args.seed_scale).replace('.','p')
if args.avoid_index_cap:stem+='_capavoid'
if args.low_force_refine:stem+='_forceopt'
if args.raised_spool:stem+='_pedestal'
if args.spool_handle:stem+='_handle'
if args.spool_handle and args.handle_radius!=.012:stem+='_radius'+str(args.handle_radius).replace('.','p')
if args.spool_handle and args.handle_clearance!=.010:stem+='_clear'+str(args.handle_clearance).replace('.','p')
if args.handle_solref_time:stem+='_solref'+str(args.handle_solref_time).replace('.','p')
(OUT/f'{stem}_screen.json').write_text(json.dumps(report,indent=2),encoding='utf8')
if eligible:
    np.savez_compressed(OUT/f'{stem}_candidate.npz',qpos=data.qpos.copy())
    (OUT/f'{stem}_variant.xml').write_text(ET.tostring(scene,encoding='unicode'),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ('joint_names','deep_contacts')}
                 | {'deep_count':len(deep)}))
