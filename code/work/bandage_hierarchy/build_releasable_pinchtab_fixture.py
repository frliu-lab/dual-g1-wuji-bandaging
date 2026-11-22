"""Add a switchable world fixture to the cloth end for handoff diagnostics."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
source=OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
root=ET.parse(source).getroot()
points=np.fromstring(root.find(".//flexcomp[@name='joint_bandage_strip']").get('point'),sep=' ').reshape(66,3)
equality=root.find('equality')
if equality is None:equality=ET.SubElement(root,'equality')
for vertex in (64,65):
    ET.SubElement(equality,'connect',{'name':f'bandage_tab_fixture_{vertex}',
        'body1':f'joint_bandage_strip_{vertex}',
        'anchor':' '.join(f'{x:.8f}' for x in points[vertex]),
        'active':'true','solref':'.005 1'})
scene=OUT/'joint_dual_g1_pinchtab_releasable_fixture.xml'
ET.ElementTree(root).write(scene,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(scene))
d=mujoco.MjData(m)
ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,f'bandage_tab_fixture_{i}') for i in (64,65)]
assert all(i>=0 for i in ids)
report={'scope':'Switchable world-to-flex-vertex fixture for handoff counterfactual; external support, not robot grasp',
        'scene':scene.name,'fixture_equality_ids':ids,'initially_active':[bool(d.eq_active[i]) for i in ids],
        'compiled_nq':m.nq,'compiled_nv':m.nv}
(OUT/'joint_dual_g1_pinchtab_releasable_fixture_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
