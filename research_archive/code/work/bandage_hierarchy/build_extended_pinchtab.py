"""Extend the flexible tab past the fingertip contact line for slip-margin tests."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
source=OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
root=ET.parse(source).getroot()
flex=root.find(".//flexcomp[@name='joint_bandage_strip']")
points=np.fromstring(flex.get('point'),sep=' ').reshape(-1,2,3)
elements=np.fromstring(flex.get('element'),sep=' ',dtype=int).reshape(-1,3)
tangent=points[-1].mean(axis=0)-points[-2].mean(axis=0)
tangent/=np.linalg.norm(tangent)
new_points=np.concatenate([points]+[points[-1:]+k*.005*tangent for k in range(1,5)],axis=0)
new_elements=elements.tolist()
for row in range(len(points),len(new_points)):
    a=2*(row-1);b=2*row
    new_elements.extend(([a,a+1,b],[a+1,b+1,b]))
flex.set('point',' '.join(f'{x:.7f}' for x in new_points.ravel()))
flex.set('element',' '.join(map(str,np.asarray(new_elements).ravel())))
scene=OUT/'joint_dual_g1_pinchtab_extended20mm.xml'
ET.ElementTree(root).write(scene,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(scene))
assert m.nflexvert==74
report={'scope':'20 mm flexible extension of measured-gap pull tab; static compile only',
        'scene':scene.name,'old_vertices':len(points)*2,'new_vertices':m.nflexvert,
        'new_end_center_m':new_points[-1].mean(axis=0).tolist()}
(OUT/'joint_dual_g1_pinchtab_extended20mm_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

