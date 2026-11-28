"""Prototype a thin rigid reinforced hem attached to the cloth free end."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
source=OUT/'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'
target=OUT/'joint_dual_g1_pinchtab_extended20mm_reinforced_hem.xml'
root=ET.parse(source).getroot()
world=root.find('worldbody')
equality=root.find('equality')
contact=root.find('contact')
strip=root.find(".//flexcomp[@name='joint_bandage_strip']")
vertices=np.fromstring(strip.get('point'),sep=' ').reshape(-1,3)[[72,73]]
center=np.mean(vertices,axis=0)
hem_center=center+np.array([0.,.004,0.])
axis=vertices[1]-vertices[0]
axis/=np.linalg.norm(axis)
half_length=.5*np.linalg.norm(vertices[1]-vertices[0])+.002
hem=ET.SubElement(world,'body',{'name':'bandage_reinforced_hem',
    'pos':' '.join(f'{x:.8f}' for x in hem_center)})
ET.SubElement(hem,'freejoint',{'name':'bandage_reinforced_hem_free'})
local0=vertices[0]-center;local1=vertices[1]-center
ET.SubElement(hem,'geom',{'name':'bandage_reinforced_hem_grasp',
    'type':'capsule','fromto':' '.join(f'{x:.8f}' for x in np.r_[local0,local1]),
    'size':'.0025','mass':'.004','friction':'1.2 .005 .0001',
    'rgba':'.92 .87 .70 1'})
for i,point in zip((72,73),vertices):
    ET.SubElement(equality,'connect',{'name':f'bandage_hem_seam_{i}',
        'body1':'bandage_reinforced_hem','body2':f'joint_bandage_strip_{i}',
        'anchor':' '.join(f'{x:.8f}' for x in point-hem_center),'solref':'.005 1'})
for i in (70,71,72,73):
    ET.SubElement(contact,'exclude',{'name':f'bandage_hem_internal_{i}',
        'body1':'bandage_reinforced_hem','body2':f'joint_bandage_strip_{i}'})
ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(target))
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
for vertex, point in zip((72,73),vertices):
    seam=m.equality(f'bandage_hem_seam_{vertex}').id
    for body,local in zip((m.eq_obj1id[seam],m.eq_obj2id[seam]),
                          m.eq_data[seam,:6].reshape(2,3)):
        np.testing.assert_allclose(d.xpos[body]+d.xmat[body].reshape(3,3)@local,
                                   point,atol=1.e-7,rtol=0.)
report={'scope':'Alternative physical material hypothesis: thin 5 mm diameter reinforced end hem, 4 g, sewn to two flex-end vertices; not measured hardware',
        'source_scene':source.name,'scene':target.name,
        'hem_center_world_m':hem_center.tolist(),'hem_endpoint_world_m':[x.tolist() for x in vertices],
        'hem_half_length_m':half_length,'compiled':True,'model_nq':m.nq,
        'seam_equalities':[f'bandage_hem_seam_{i}' for i in (72,73)]}
(OUT/'joint_bandage_reinforced_hem_build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
