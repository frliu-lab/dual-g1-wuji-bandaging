"""Inspect actual MuJoCo contact forces at the failed prototype-handle lift endpoint."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
source=next(p for p in OUT.glob('joint_support_pinch_staged_wbc_report_spool_settled_handle_*.json')
            if (r:=json.loads(p.read_text(encoding='utf8'))).get('b_lift_force_goal_n')==1.5)
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
state=np.load(source.with_name(source.stem+'_final_state.npz'))
data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel']
mujoco.mj_forward(model,data)
rows=[]
for i,c in enumerate(data.contact):
    if c.geom1<0 or c.geom2<0:continue
    geom_names=(model.geom(c.geom1).name,model.geom(c.geom2).name)
    if 'bandage_handle_prototype' not in geom_names:continue
    wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
    rows.append({'geoms':geom_names,'body_names':(model.body(model.geom_bodyid[c.geom1]).name,
                                                   model.body(model.geom_bodyid[c.geom2]).name),
                 'normal_n':float(wrench[0]),
                 'force_world_n':(c.frame.reshape(3,3).T@wrench[:3]).tolist(),
                 'friction':float(c.friction[0])})
result={'scope':'Recomputed endpoint forces from saved qpos and qvel; qualitative diagnostic, not exact historic contact impulses',
        'source':source.name,'wrist_z_m':float(data.xpos[model.body('g1_b/wuji_left/l_wrist').id][2]),
        'spool_z_m':float(data.xpos[model.body('bandage_spool').id][2]),
        'spool_mass_kg':float(model.body_subtreemass[model.body('bandage_spool').id]),
        'contacts':rows}
(OUT/'joint_spool_lift_contact_force_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result))
