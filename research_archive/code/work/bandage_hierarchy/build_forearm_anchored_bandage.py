"""Attach bandage start vertices to the simulated forearm instead of world."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
source=OUT/'joint_dual_g1_pinchtab_extended20mm_reinforced_hem.xml'
target=OUT/'joint_dual_g1_pinchtab_extended20mm_reinforced_hem_forearm_anchor.xml'
root=ET.parse(source).getroot()
flex=root.find(".//flexcomp[@name='joint_bandage_strip']")
point=np.fromstring(flex.get('point'),sep=' ').reshape(-1,3)
pin=flex.find('pin')
if pin is None or pin.get('id')!='0 1':
    raise RuntimeError('Expected bandage start pinned to world at vertices 0 and 1')
flex.remove(pin)
equality=root.find('equality')
for i in (0,1):
    ET.SubElement(equality,'connect',{'name':f'bandage_forearm_anchor_{i}',
        'body1':f'joint_bandage_strip_{i}','body2':'training_forearm_dynamic',
        'anchor':' '.join(f'{x:.8f}' for x in point[i]),'solref':'.005 1'})
ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(target))
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
report={'scope':'Alternative anchoring hypothesis: cloth start sewn to movable forearm instead of world; no hardware measurement',
        'source_scene':source.name,'scene':target.name,'compiled':True,
        'start_vertices_world_m':point[:2].tolist(),'model_nq':m.nq,
        'anchor_equalities':['bandage_forearm_anchor_0','bandage_forearm_anchor_1']}
(OUT/'joint_bandage_forearm_anchor_build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
