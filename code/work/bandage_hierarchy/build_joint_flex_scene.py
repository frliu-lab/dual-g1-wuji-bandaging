"""Compile a deformable prewrapped strip in the dual-G1 scene; no robot wrap claim."""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--world-anchored',action='store_true')
parser.add_argument('--radius',type=float,default=.052)
parser.add_argument('--phase-deg',type=float,default=0.)
parser.add_argument('--thickness-mm',type=float,default=2.)
parser.add_argument('--tab-outreach',type=float,default=0.)
parser.add_argument('--edge-damping',type=float,default=.2)
args=parser.parse_args()
if args.thickness_mm<=0 or args.tab_outreach<0 or args.edge_damping<0:
    parser.error('Thickness must be positive; tab outreach and damping nonnegative')
screen=json.loads((OUT/'joint_b_reposition_static_screen.json').read_text())
path=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
root=ET.parse(OUT/screen['scene']).getroot()
ET.SubElement(root.find('worldbody'),'geom',{'name':'joint_bandage_flex_contact_marker',
    'type':'sphere','pos':'1000 1000 1000','size':'.001','contype':'0','conaffinity':'0'})
forearm=root.find(".//body[@name='training_forearm_dynamic']")
assert forearm is not None
base_model=mujoco.MjModel.from_xml_path(str(OUT/screen['scene']))
base_data=mujoco.MjData(base_model)
seed=np.asarray(path['samples'][-1]['qpos'])
base_data.qpos[:len(seed)]=seed
mujoco.mj_forward(base_model,base_data)
forearm_body_id=base_model.body('training_forearm_dynamic').id
forearm_origin=base_data.xpos[forearm_body_id].copy()
forearm_rotation=base_data.xmat[forearm_body_id].reshape(3,3).copy()
axis=np.array([.34,-.17,-.04],dtype=float)
axis/=np.linalg.norm(axis)
radial_a=np.cross(axis,np.array([0.,0.,1.]))
radial_a/=np.linalg.norm(radial_a)
radial_b=np.cross(axis,radial_a)
center=np.array([.17,-.085,-.02])
points=[];triangles=[]
for i in range(33):
    u=i/32
    theta=4*np.pi*u+np.deg2rad(args.phase_deg)
    for side in (-.5,.5):
        vertex=(center+.11*(u-.5)*axis+.018*side*axis+
                args.radius*(np.cos(theta)*radial_a+np.sin(theta)*radial_b))
        if args.world_anchored:
            vertex=forearm_origin+forearm_rotation@vertex
            vertex+=args.tab_outreach*np.clip((i-25)/7,0,1)*np.array([-.03,.055,-.025])
        points.extend(vertex)
    if i:
        a=2*(i-1);b=2*i
        triangles.extend((a,a+1,b,a+1,b+1,b))
flex_parent=root.find('worldbody') if args.world_anchored else forearm
flex=ET.SubElement(flex_parent,'flexcomp',{'name':'joint_bandage_strip','type':'direct',
    'dim':'2','mass':'.025','radius':str(args.thickness_mm/1000),'rgba':'.95 .9 .75 1',
    'point':' '.join(f'{x:.7f}' for x in points),
    'element':' '.join(map(str,triangles))})
ET.SubElement(flex,'pin',{'id':'0 1'})
ET.SubElement(flex,'edge',{'equality':'true','damping':str(args.edge_damping)})
ET.SubElement(flex,'contact',{'solref':'.0015 1','friction':'.8 .005 .0001',
                              'selfcollide':'none'})
variant=(f'_radius{args.radius*1000:g}mm_phase{args.phase_deg:g}deg'
         if args.radius!=.052 or args.phase_deg else '')
if args.thickness_mm!=2.:
    variant+=f'_thickness{args.thickness_mm:g}mm'
if args.tab_outreach:
    variant+=f'_tab{args.tab_outreach:g}'
if args.edge_damping!=.2:
    variant+=f'_damping{args.edge_damping:g}'
xml_path=OUT/f'joint_dual_g1_prewrapped_flex_scene{"_world_anchored" if args.world_anchored else ""}{variant}.xml'
ET.ElementTree(root).write(xml_path,encoding='utf8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(xml_path))
data=mujoco.MjData(model)
data.qpos[:len(seed)]=seed
mujoco.mj_forward(model,data)
flex_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_FLEX,'joint_bandage_strip')
forearm_id=model.geom('manikin_forearm').id
contacts=[]
for c in data.contact:
    if flex_id not in c.flex:
        continue
    geom=int(c.geom1 if c.geom1>=0 else c.geom2)
    contacts.append({'geom':geom,'geom_name':model.geom(geom).name if geom>=0 else None,
                     'body_name':model.body(model.geom_bodyid[geom]).name if geom>=0 else None,
                     'is_forearm':forearm_id in c.geom,
                     'penetration_m':max(0.,-float(c.dist))})
report={'scope':'Dual G1 plus prewrapped deformable strip static compile only; strip is not robot-wrapped',
        'world_anchored':args.world_anchored,
        'strip_radius_m':args.radius,'strip_phase_deg':args.phase_deg,
        'strip_collision_thickness_mm':args.thickness_mm,
        'tab_outreach':args.tab_outreach,
        'edge_damping':args.edge_damping,
        'scene':xml_path.name,'nflex':model.nflex,'nflexvert':model.nflexvert,
        'robot_seed_qpos_length':len(seed),'compiled_nq':model.nq,
        'initial_flex_contacts':len(contacts),
        'initial_flex_forearm_contacts':sum(c['is_forearm'] for c in contacts),
        'initial_flex_max_penetration_m':max((c['penetration_m'] for c in contacts),default=0.),
        'deepest_flex_contacts':sorted(contacts,key=lambda c:c['penetration_m'],reverse=True)[:10],
        'flex_contacts_have_negative_geom_id':any(any(g<0 for g in c.geom)
            for c in data.contact if flex_id in c.flex),
        'finite_state':bool(np.all(np.isfinite(data.qpos)))}
(OUT/f'joint_dual_g1_prewrapped_flex_scene_check{"_world_anchored" if args.world_anchored else ""}{variant}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
