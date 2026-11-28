"""Parent a pinned flex bandage under the moving forearm body."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
source=OUT/'joint_dual_g1_pinchtab_extended20mm_reinforced_hem.xml'
target=OUT/'joint_dual_g1_pinchtab_extended20mm_reinforced_hem_body_parented_refaligned.xml'
m0=mujoco.MjModel.from_xml_path(str(source));d0=mujoco.MjData(m0)
d0.qpos[:167]=np.load(OUT/'joint_b_pinchtab_rrt_path.npz')['qpos'][-1,:167]
mujoco.mj_forward(m0,d0)
forearm_id=m0.body('training_forearm_dynamic').id
origin=d0.xpos[forearm_id].copy()
R=d0.xmat[forearm_id].reshape(3,3).copy()
elbow_seed=float(d0.qpos[m0.joint('training_elbow_hinge').qposadr[0]])
root=ET.parse(source).getroot()
world=root.find('worldbody')
flex=world.find("flexcomp[@name='joint_bandage_strip']")
if flex is None or flex.find('pin') is None or flex.find('pin').get('id')!='0 1':
    raise RuntimeError('Expected world-owned bandage with first two vertices pinned')
point=np.fromstring(flex.get('point'),sep=' ').reshape(-1,3)
local=(R.T@(point-origin).T).T
flex.set('point',' '.join(f'{x:.8f}' for x in local.ravel()))
world.remove(flex)
forearm=world.find(".//body[@name='training_forearm_dynamic']")
if forearm is None:raise RuntimeError('Missing simulated forearm body')
forearm.set('quat',f'{np.cos(elbow_seed/2):.10f} 0 {np.sin(elbow_seed/2):.10f} 0')
forearm.find("joint[@name='training_elbow_hinge']").set('ref',f'{elbow_seed:.10f}')
forearm.append(flex)
ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(target))
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
report={'scope':'Alternative bandage origin attached to moving forearm parent; coordinates transformed at recorded seed elbow state; no hardware evidence',
        'scene':target.name,'source_scene':source.name,'source_nq':m0.nq,'new_nq':m.nq,
        'seed_forearm_origin_world_m':origin.tolist(),'seed_forearm_rotation_world':R.tolist(),
        'elbow_reference_rad':elbow_seed,
        'first_two_original_world_m':point[:2].tolist(),'compiled':True}
(OUT/'joint_bandage_body_parented_refaligned_build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
