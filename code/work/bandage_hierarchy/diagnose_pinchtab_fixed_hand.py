"""Separate flexible-tab relaxation from robot controller motion."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
scene=OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
root=ET.parse(scene).getroot()
seed=json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
reach=json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
hand=json.loads((OUT/'joint_b_natural_pinch_workspace_screen.json').read_text())
candidate=next(row for row in hand['results'] if row['base']=='reach' and row['trial']==4)
cases=[]
for terminal_pinned in (False,True):
    variant=ET.fromstring(ET.tostring(root))
    if terminal_pinned:
        flex=variant.find(".//flexcomp[@name='joint_bandage_strip']")
        ET.SubElement(flex,'pin',{'id':'64 65'})
    m=mujoco.MjModel.from_xml_string(ET.tostring(variant,encoding='unicode'))
    d=mujoco.MjData(m)
    q=d.qpos.copy();q[:len(seed['samples'][-1]['qpos'])]=seed['samples'][-1]['qpos']
    q[[m.joint(n).qposadr[0] for n in reach['joint_names']]]=reach['results'][0]['joint_targets_rad']
    q[[m.joint(n).qposadr[0] for n in hand['joint_names']]]=candidate['joint_targets_rad']
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    edge0=d.flexvert_xpos[64:66].mean(axis=0).copy()
    first_flex_dof=next(int(m.jnt_dofadr[j]) for j in range(m.njnt)
                        if int(m.jnt_qposadr[j])>=len(seed['samples'][-1]['qpos']))
    opposed_steps=0;max_pen=0.;trace=[]
    steps=round(.5/m.opt.timestep)
    for i in range(steps):
        d.qpos[:len(seed['samples'][-1]['qpos'])]=q[:len(seed['samples'][-1]['qpos'])]
        d.qvel[:first_flex_dof]=0.
        mujoco.mj_step(m,d)
        normal={'thumb':[],'index':[]}
        for c in d.contact:
            geoms=[int(g) for g in c.geom if g>=0]
            bodies=[m.body(m.geom_bodyid[g]).name for g in geoms]
            if c.dist<0 and any(n.startswith('g1_b/') for n in bodies) and 'floor' not in [m.geom(g).name for g in geoms]:
                max_pen=max(max_pen,-float(c.dist))
            if not any(f>=0 for f in c.flex) or c.dist>=.003:continue
            if 'g1_b/wuji_right/r_thumb_tip_sensor_frame' in bodies:normal['thumb'].append(c.frame[:3].copy())
            if 'g1_b/wuji_right/r_index_finger_tip_sensor_frame' in bodies:normal['index'].append(c.frame[:3].copy())
        opposed=any(np.dot(a,b)<-.5 for a in normal['thumb'] for b in normal['index'])
        opposed_steps+=opposed
        if (i+1)%round(.02/m.opt.timestep)==0:
            trace.append({'time_s':float(d.time),'opposed_contact':bool(opposed),
                          'edge_drift_m':float(np.linalg.norm(d.flexvert_xpos[64:66].mean(axis=0)-edge0))})
    cases.append({'terminal_pinned_to_world':terminal_pinned,
                  'opposed_tip_contact_fraction':opposed_steps/steps,
                  'max_b_nonfloor_penetration_m':max_pen,
                  'max_free_edge_drift_m':max(t['edge_drift_m'] for t in trace),
                  'trace':trace})
report={'scope':'Externally fixed original bodies, with or without world-pinned strip end; diagnostic only, not robot execution',
        'duration_s':.5,'cases':cases}
(OUT/'joint_b_pinchtab_fixed_hand_relaxation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in c.items() if k!='trace'} for c in cases]))
