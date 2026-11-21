"""Find a two-digit static contact seed on a rigid bandage-end proxy."""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--fine',action='store_true')
parser.add_argument('--radius',type=float,default=.009)
args=parser.parse_args()
screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
path=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
root=ET.parse(OUT/screen['scene']).getroot()
world=root.find('worldbody')
ET.SubElement(world,'geom',{'name':'bandage_tail_static_proxy','type':'cylinder',
                            'pos':'.05 .166 .79','quat':'.70710678 0 .70710678 0',
                            'size':f'{args.radius} .025','rgba':'.95 .89 .70 1',
                            'friction':'1 .005 .0001'})
xml=ET.tostring(root,encoding='unicode')
model=mujoco.MjModel.from_xml_string(xml)
data=mujoco.MjData(model)
q=np.asarray(path['samples'][-1]['qpos'])
geom=model.geom('bandage_tail_static_proxy').id
thumb=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/wuji_right/r_thumb_') and 'abd' not in model.joint(j).name]
index=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/wuji_right/r_index_') and 'abd' not in model.joint(j).name]
rows=[]
xs=(.05,.055) if args.fine else (.045,.05,.055)
ys=np.linspace(.151,.159,5) if args.fine else (.155,.165,.175)
zs=np.linspace(.792,.799,8) if args.fine else (.775,.785,.795)
thumb_flexes=np.linspace(.45,.55,5) if args.fine else (.2,.35,.5)
index_flexes=np.linspace(.30,.40,5) if args.fine else (.2,.35,.5)
for x in xs:
 for y in ys:
  for z in zs:
   model.geom_pos[geom]=[x,y,z]
   for thumb_flex in thumb_flexes:
    for index_flex in index_flexes:
     data.qpos[:]=q
     for ids,flex in ((thumb,thumb_flex),(index,index_flex)):
      for j in ids:
       data.qpos[model.jnt_qposadr[j]]=np.clip(flex,model.jnt_range[j,0]+.01,
                                               model.jnt_range[j,1]-.01)
     mujoco.mj_forward(model,data)
     hits={'thumb':[],'index':[]};worst_pen=0.;other_deep=[]
     for c in data.contact:
      names=[model.geom(g).name for g in c.geom]
      bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom]
      if c.dist<0 and 'floor' not in names and any(n.startswith('g1_b/') for n in bodies):
       worst_pen=max(worst_pen,-float(c.dist))
      if geom not in c.geom:
       continue
      for digit in hits:
       if any(n.startswith(f'g1_b/wuji_right/r_{digit}_') for n in bodies):
        # Contact frame normal points from geom1 to geom2.
        normal=c.frame[:3].copy()*(1 if c.geom1==geom else -1)
        hits[digit].append((float(c.dist),normal))
      if c.dist<-.003 and not any(n.startswith('g1_b/wuji_right/r_') for n in bodies):
       other_deep.append(bodies)
     thumb_hit=min((h[0] for h in hits['thumb']),default=None)
     index_hit=min((h[0] for h in hits['index']),default=None)
     opposing=min((float(np.dot(a[1],b[1])) for a in hits['thumb'] for b in hits['index']),default=None)
     good=(thumb_hit is not None and index_hit is not None and
           -.003<=thumb_hit<=.001 and -.003<=index_hit<=.001 and
           opposing is not None and opposing<-.5 and worst_pen<.003 and not other_deep)
     rows.append({'proxy_pos_m':[x,y,z],'thumb_flex_rad':thumb_flex,
                  'index_flex_rad':index_flex,'thumb_min_contact_distance_m':thumb_hit,
                  'index_min_contact_distance_m':index_hit,
                  'minimum_thumb_index_contact_normal_dot':opposing,
                  'maximum_b_nonfloor_penetration_m':worst_pen,
                  'two_digit_opposing_static_seed':good})
good=[r for r in rows if r['two_digit_opposing_static_seed']]
report={'scope':'Fixed rigid proxy geometry seed only; no flexible strip, object motion, forces, or grasp hold',
        'proxy_shape':f'Cylinder radius {args.radius*1000:g} mm, half-length 25 mm; axis along world x',
        'fine_search':args.fine,
        'candidates':len(rows),'passing_two_digit_seeds':len(good),
        'best_candidates':sorted(good,key=lambda r:r['maximum_b_nonfloor_penetration_m'])[:20],
        'nearest_candidates':sorted(rows,key=lambda r:(r['thumb_min_contact_distance_m'] is None or
                     r['index_min_contact_distance_m'] is None,
                     r['maximum_b_nonfloor_penetration_m']))[:20]}
suffix=f'_{round(args.radius*1000):g}mm' if args.radius!=.009 else ''
(OUT/(f'joint_b_tail_static_proxy_contact_fine_screen{suffix}.json' if args.fine else
      f'joint_b_tail_static_proxy_contact_screen{suffix}.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('best_candidates','nearest_candidates')}|
                 {'best':report['best_candidates'][:3]}))
