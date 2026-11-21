"""Screen rigid tail orientation for opposed thumb-tip and index-tip contact."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
path=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q=np.asarray(path['samples'][-1]['qpos'])
base_model=mujoco.MjModel.from_xml_path(str(OUT/screen['scene']))
base_data=mujoco.MjData(base_model)
base_data.qpos[:]=q
for j in range(base_model.njnt):
 if base_model.joint(j).name.startswith('g1_b/wuji_right/r_index_') and 'abd' not in base_model.joint(j).name:
  base_data.qpos[base_model.jnt_qposadr[j]]=.325
mujoco.mj_forward(base_model,base_data)
thumb_base=base_model.body('g1_b/wuji_right/r_thumb_tip_sensor_frame').id
index_base=base_model.body('g1_b/wuji_right/r_index_finger_tip_sensor_frame').id
thumb_base_geom=next(i for i in range(base_model.ngeom) if base_model.geom_bodyid[i]==thumb_base)
index_base_geom=next(i for i in range(base_model.ngeom) if base_model.geom_bodyid[i]==index_base)
axis=base_data.geom_xpos[thumb_base_geom]-base_data.geom_xpos[index_base_geom]
axis/=np.linalg.norm(axis)
rotation,_=Rotation.align_vectors([axis],[[0,0,1]])
xyzw=rotation.as_quat()
quat=np.r_[xyzw[3],xyzw[:3]]
root=ET.parse(OUT/screen['scene']).getroot()
ET.SubElement(root.find('worldbody'),'geom',{
    'name':'bandage_tail_tip_pinch_proxy','type':'cylinder','pos':'.05 .166 .785',
    'quat':' '.join(map(str,quat)),
    'size':'.007 .025','rgba':'.95 .89 .70 1','friction':'1 .005 .0001'})
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
data=mujoco.MjData(model)
thumb_id=model.body('g1_b/wuji_right/r_thumb_tip_sensor_frame').id
index_id=model.body('g1_b/wuji_right/r_index_finger_tip_sensor_frame').id
thumb_js=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/wuji_right/r_thumb_') and 'abd' not in model.joint(j).name]
index_js=[j for j in range(model.njnt) if model.joint(j).name.startswith('g1_b/wuji_right/r_index_') and 'abd' not in model.joint(j).name]
geom_id=model.geom('bandage_tail_tip_pinch_proxy').id
rows=[]
for x in (.03,.035,.04,.045):
 for y in np.linspace(.15,.17,6):
  for z in np.linspace(.768,.790,7):
   model.geom_pos[geom_id]=[x,y,z]
   for tf in (.4,.45,.5,.55):
    for inf in (.25,.30,.325,.35,.40):
     data.qpos[:]=q
     for joints,flex in ((thumb_js,tf),(index_js,inf)):
      for j in joints:
       data.qpos[model.jnt_qposadr[j]]=np.clip(flex,model.jnt_range[j,0]+.01,model.jnt_range[j,1]-.01)
     mujoco.mj_forward(model,data)
     normals={'thumb':[],'index':[]};worst=0.
     for c in data.contact:
      names=[model.geom(g).name for g in c.geom]
      bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom]
      if c.dist<0 and 'floor' not in names and any(n.startswith('g1_b/') for n in bodies):
       worst=max(worst,-float(c.dist))
      if geom_id not in c.geom:continue
      for digit,bid in (('thumb',thumb_id),('index',index_id)):
       if model.body(bid).name in bodies:
        normals[digit].append((float(c.dist),c.frame[:3].copy()*
                              (1 if c.geom1==geom_id else -1)))
     thumb_dist=min((v[0] for v in normals['thumb']),default=None)
     index_dist=min((v[0] for v in normals['index']),default=None)
     dot=min((float(np.dot(a[1],b[1])) for a in normals['thumb'] for b in normals['index']),default=None)
     passed=(thumb_dist is not None and index_dist is not None and
             -.003<=thumb_dist<=.001 and -.003<=index_dist<=.001 and
             dot is not None and dot<-.5 and worst<.003)
     rows.append({'proxy_center_m':[float(x),float(y),float(z)],
                  'proxy_quaternion_wxyz':quat.tolist(),
                  'thumb_flex_rad':tf,'index_flex_rad':inf,
                  'thumb_tip_distance_m':thumb_dist,'index_tip_distance_m':index_dist,
                  'minimum_opposing_normal_dot':dot,'max_b_nonfloor_penetration_m':worst,
                  'two_fingertip_seed_pass':passed})
passed=[r for r in rows if r['two_fingertip_seed_pass']]
report={'scope':'Fixed rigid proxy fingertip geometry only; no grasp dynamics',
        'axis_world':axis.tolist(),'candidates':len(rows),'passing_seeds':len(passed),
        'both_tip_contact_candidates':sum(r['thumb_tip_distance_m'] is not None and
                                          r['index_tip_distance_m'] is not None for r in rows),
        'thumb_tip_contact_candidates':sum(r['thumb_tip_distance_m'] is not None for r in rows),
        'index_tip_contact_candidates':sum(r['index_tip_distance_m'] is not None for r in rows),
        'closest_both_tip_candidates':sorted([r for r in rows if r['thumb_tip_distance_m'] is not None and
                                                r['index_tip_distance_m'] is not None],
                key=lambda r:r['max_b_nonfloor_penetration_m'])[:20],
        'best_candidates':sorted(passed,key=lambda r:max(abs(r['thumb_tip_distance_m']),
                                                       abs(r['index_tip_distance_m'])))[:20]}
(OUT/'joint_b_tail_tip_pinch_static_screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='best_candidates'}|
                 {'best':report['best_candidates'][:3]}))
