"""Route a flexible pull tab around the index middle phalanx."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
source=OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab.xml'
root=ET.parse(source).getroot()
flex=root.find(".//flexcomp[@name='joint_bandage_strip']")
original=np.fromstring(flex.get('point'),sep=' ').reshape(33,2,3)
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
reach=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
hand=json.loads((OUT/'joint_b_natural_pinch_workspace_screen.json').read_text())
candidate=next(row for row in hand['results'] if row['base']=='reach' and row['trial']==4)
rows=[]
best=None
for dx in (0.,.005,.01,.015):
    for dy in (-.015,-.0075,0.,.0075,.015):
      for dz in (.015,.02,.025):
        p=original.copy()
        for i in range(26,32):
            t=(i-25)/7
            p[i]+=np.sin(np.pi*t)*np.array([dx,dy,dz])
        flex.set('point',' '.join(f'{v:.7f}' for v in p.ravel()))
        model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        d=mujoco.MjData(model)
        q=d.qpos.copy();q[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
        q[[model.joint(n).qposadr[0] for n in reach['joint_names']]]=reach['results'][0]['joint_targets_rad']
        q[[model.joint(n).qposadr[0] for n in hand['joint_names']]]=candidate['joint_targets_rad']
        d.qpos[:]=q;mujoco.mj_forward(model,d)
        normals={'thumb':[],'index':[]};max_pen=0.;worst=None
        for c in d.contact:
            geoms=[int(g) for g in c.geom if g>=0]
            bodies=[model.body(model.geom_bodyid[g]).name for g in geoms]
            if c.dist<0 and any(n.startswith('g1_b/') for n in bodies) and 'floor' not in [model.geom(g).name for g in geoms]:
                if -float(c.dist)>max_pen:
                    max_pen=-float(c.dist);worst=bodies
            if not any(f>=0 for f in c.flex) or c.dist>=.003:continue
            if any('r_thumb_tip_sensor_frame' in n for n in bodies):normals['thumb'].append(c.frame[:3].copy())
            if any('r_index_finger_tip_sensor_frame' in n for n in bodies):normals['index'].append(c.frame[:3].copy())
        best_dot=min((float(np.dot(a,b)) for a in normals['thumb'] for b in normals['index']),default=None)
        row={'bulge_x_m':dx,'bulge_y_m':dy,'bulge_z_m':dz,'max_b_nonfloor_penetration_m':max_pen,
             'worst_bodies':worst,'thumb_tip_contacts':len(normals['thumb']),
             'index_tip_contacts':len(normals['index']),'best_normal_dot':best_dot,
             'static_contact_gate_pass':bool(best_dot is not None and best_dot<-.5 and max_pen<.003)}
        rows.append(row)
        if row['static_contact_gate_pass'] and (best is None or max_pen<best['max_b_nonfloor_penetration_m']):
            best=row.copy()
            target=OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
            ET.ElementTree(root).write(target,encoding='utf8',xml_declaration=True)
rows.sort(key=lambda r:(not r['static_contact_gate_pass'],
                        r['max_b_nonfloor_penetration_m'],
                        1 if r['best_normal_dot'] is None else r['best_normal_dot']))
report={'scope':'Static cloth-tab path-shape sweep with fixed hand candidate; no force closure or motion',
        'cases':rows,'best_pass':best}
(OUT/'joint_b_pinchtab_bulge_sweep.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passing':sum(r['static_contact_gate_pass'] for r in rows),'best':best,'top':rows[:5]}))
