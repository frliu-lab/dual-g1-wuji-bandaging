"""Static B-right arm/digit reach screen for the actual flexible strip free edge."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--ignore-collision-in-objective',action='store_true')
parser.add_argument('--all-digits',action='store_true')
parser.add_argument('--staged-from-unconstrained',action='store_true')
parser.add_argument('--phase-deg',type=float,default=225.)
parser.add_argument('--tab-outreach',type=float,default=0.)
parser.add_argument('--index-row',type=int,default=32)
parser.add_argument('--surface-offset-mm',type=float,default=4.)
parser.add_argument('--collision-cap-mm',type=float,default=3.)
args=parser.parse_args()
if not 26<=args.index_row<=32:
    parser.error('Index row must lie on the loose tab, 26..32')
tab_suffix=f'_tab{args.tab_outreach:g}' if args.tab_outreach else ''
model=mujoco.MjModel.from_xml_path(str(OUT/f'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase{args.phase_deg:g}deg_thickness1mm{tab_suffix}.xml'))
data=mujoco.MjData(model)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
robot_q=np.asarray(seed['samples'][-1]['qpos'])
q0=data.qpos.copy();q0[:len(robot_q)]=robot_q
data.qpos[:]=q0;mujoco.mj_forward(model,data)
vertices=data.flexvert_xpos[:66].reshape(33,2,3)
centers=vertices.mean(axis=1)
edge=centers[-1].copy()
def row_normal(row):
    tangent=centers[row]-centers[row-1]
    width=vertices[row,1]-vertices[row,0]
    normal=np.cross(tangent,width)
    return normal/np.linalg.norm(normal)
normal=row_normal(32)
index_normal=row_normal(args.index_row)
tip_names=('g1_b/wuji_right/r_thumb_tip_sensor_frame',
           'g1_b/wuji_right/r_index_finger_tip_sensor_frame')
tip_ids=[model.body(name).id for name in tip_names]
names=[model.joint(j).name for j in range(model.njnt) if
       model.joint(j).name.startswith('g1_b/right_') and
       any(key in model.joint(j).name for key in ('shoulder','elbow','wrist'))]
names += [model.joint(j).name for j in range(model.njnt) if
          model.joint(j).name.startswith(('g1_b/wuji_right/r_thumb_',
                                          'g1_b/wuji_right/r_index_')) and
          'abd' not in model.joint(j).name]
if args.all_digits:
    names += [model.joint(j).name for j in range(model.njnt) if
              model.joint(j).name.startswith(('g1_b/wuji_right/r_middle_',
                                              'g1_b/wuji_right/r_ring_',
                                              'g1_b/wuji_right/r_pinky_'))]
joint_ids=np.array([model.joint(name).id for name in names])
qadr=np.array([model.jnt_qposadr[j] for j in joint_ids])
start=q0[qadr].copy()
lo=model.jnt_range[joint_ids,0]+1e-4
hi=model.jnt_range[joint_ids,1]-1e-4
rows=[]
for sign in (1.,-1.):
    offset=args.surface_offset_mm/1000
    goals=np.array([edge+sign*offset*normal,
                    centers[args.index_row]-sign*offset*index_normal])
    def evaluate(x):
        data.qpos[:]=q0;data.qpos[qadr]=x
        mujoco.mj_forward(model,data)
        tips=np.array([data.xpos[i] for i in tip_ids])
        max_pen=0.;robot_flex_pen=0.
        for c in data.contact:
            rigid=[int(g) for g in c.geom if g>=0]
            bodies=[model.body(model.geom_bodyid[g]).name for g in rigid]
            if any(name.startswith('g1_b/') for name in bodies):
                if any(f>=0 for f in c.flex):
                    robot_flex_pen=max(robot_flex_pen,-float(c.dist))
                elif 'floor' not in [model.geom(g).name for g in rigid]:
                    max_pen=max(max_pen,-float(c.dist))
        return tips,max_pen,robot_flex_pen
    def residual(x):
        tips,max_pen,robot_flex_pen=evaluate(x)
        cap=args.collision_cap_mm/1000
        return np.r_[.2*(tips-goals).ravel(),
                     (0. if args.ignore_collision_in_objective else 10.)*max(0.,max_pen-cap),
                     (0. if args.ignore_collision_in_objective else 10.)*max(0.,robot_flex_pen-cap),
                     .001*(x-start)]
    fit_start=start
    if args.staged_from_unconstrained:
        source=OUT/f'joint_b_flex_free_edge_reach_screen_ignore_collision{"_all_digits" if args.all_digits else ""}{"_phase"+str(args.phase_deg).replace(".","p") if args.phase_deg!=225. else ""}{tab_suffix}{"_indexrow"+str(args.index_row) if args.index_row!=32 else ""}{"_offset"+str(args.surface_offset_mm).replace(".","p") if args.surface_offset_mm!=4. else ""}.json'
        previous=json.loads(source.read_text())
        assert previous['joint_names']==names
        fit_start=np.asarray(next(row['joint_targets_rad'] for row in previous['results']
                                  if row['side_sign']==sign))
    fit=least_squares(residual,np.clip(fit_start,lo,hi),bounds=(lo,hi),max_nfev=200)
    tips,max_pen,robot_flex_pen=evaluate(fit.x)
    deep=[]
    fingertip_flex_contacts={'thumb':[],'index':[]}
    for c in data.contact:
        if any(f>=0 for f in c.flex):
            contact_bodies=[model.body(model.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
            for digit,token in (('thumb','r_thumb_tip_sensor_frame'),
                                ('index','r_index_finger_tip_sensor_frame')):
                if any(token in name for name in contact_bodies) and c.dist<.003:
                    fingertip_flex_contacts[digit].append(c.frame[:3].copy())
        if c.dist>=-.003:continue
        geoms=[int(g) for g in c.geom if g>=0]
        bodies=[model.body(model.geom_bodyid[g]).name for g in geoms]
        if any(name.startswith('g1_b/') for name in bodies) and 'floor' not in [model.geom(g).name for g in geoms]:
            deep.append({'bodies':bodies,'flex':bool(any(f>=0 for f in c.flex)),
                         'penetration_m':-float(c.dist)})
    errors=np.linalg.norm(tips-goals,axis=1)
    opposed=any(np.dot(a,b)<-.5 for a in fingertip_flex_contacts['thumb']
                for b in fingertip_flex_contacts['index'])
    rows.append({'side_sign':sign,'optimizer_success':bool(fit.success),
                 'target_m':goals.tolist(),'tip_m':tips.tolist(),
                 'tip_errors_m':errors.tolist(),'max_nonfloor_robot_penetration_m':max_pen,
                 'max_robot_flex_penetration_m':robot_flex_pen,
                 'thumb_tip_flex_contact_count':len(fingertip_flex_contacts['thumb']),
                 'index_tip_flex_contact_count':len(fingertip_flex_contacts['index']),
                 'opposed_tip_flex_normals':bool(opposed),
                 'worst_contacts':sorted(deep,key=lambda c:c['penetration_m'],reverse=True)[:8],
                 'static_screen_pass':bool(max(errors)<.01 and max(max_pen,robot_flex_pen)<.003 and opposed),
                 'reach_only_screen_pass':bool(max(errors)<.01 and max(max_pen,robot_flex_pen)<.003),
                 'joint_targets_rad':fit.x.tolist()})
report={'scope':'Static B-right reach to flexible free edge; no motion trajectory or grasp force',
        'collision_ignored_in_objective':args.ignore_collision_in_objective,
        'all_digits':args.all_digits,
        'staged_from_unconstrained':args.staged_from_unconstrained,
        'phase_deg':args.phase_deg,
        'tab_outreach':args.tab_outreach,
        'index_row':args.index_row,
        'surface_offset_mm':args.surface_offset_mm,
        'collision_cap_mm':args.collision_cap_mm,
        'edge_center_m':edge.tolist(),'edge_surface_normal':normal.tolist(),
        'index_row_center_m':centers[args.index_row].tolist(),
        'joint_names':names,'results':rows}
(OUT/f'joint_b_flex_free_edge_reach_screen{"_ignore_collision" if args.ignore_collision_in_objective else ""}{"_all_digits" if args.all_digits else ""}{"_staged" if args.staged_from_unconstrained else ""}{"_phase"+str(args.phase_deg).replace(".","p") if args.phase_deg!=225. else ""}{tab_suffix}{"_indexrow"+str(args.index_row) if args.index_row!=32 else ""}{"_offset"+str(args.surface_offset_mm).replace(".","p") if args.surface_offset_mm!=4. else ""}{"_cap"+str(args.collision_cap_mm).replace(".","p") if args.collision_cap_mm!=3. else ""}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'edge_center_m':edge.tolist(),'results':[
    {k:v for k,v in row.items() if k not in ('joint_targets_rad','worst_contacts')}|
    {'worst_contacts':row['worst_contacts'][:3]} for row in rows]}))
