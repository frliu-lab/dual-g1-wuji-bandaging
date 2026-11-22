"""Isolate cloth free-edge drift from robot control and edge damping."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/wuji_multiagent/bandage_v0'
scene = OUT/'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5.xml'
seed = json.loads((OUT/'joint_b_planted_transition_synchronous_refined_lowered10mm_kinematic_audit.json').read_text())
q_robot = np.asarray(seed['samples'][-1]['qpos'])
cases = []
for damping in (.2, 2., 20.):
    for pin_end in (False, True):
        root = ET.parse(scene).getroot()
        flex = root.find(".//flexcomp[@name='joint_bandage_strip']")
        flex.find('edge').set('damping',str(damping))
        if pin_end:
            ET.SubElement(flex,'pin',{'id':'64 65'})
        m = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        d = mujoco.MjData(m)
        d.qpos[:len(q_robot)] = q_robot
        mujoco.mj_forward(m,d)
        edge0 = d.flexvert_xpos[64:66].mean(axis=0).copy()
        row28_0 = d.flexvert_xpos[56:58].mean(axis=0).copy()
        first_flex_dof = next(int(m.jnt_dofadr[j]) for j in range(m.njnt)
                              if int(m.jnt_qposadr[j]) >= len(q_robot))
        trace = []
        max_robot_pen = 0.
        for step in range(round(.5/m.opt.timestep)):
            # A diagnostic fixture holds all original scene bodies fixed;
            # only the flex vertex bodies remain dynamically integrated.
            d.qpos[:len(q_robot)] = q_robot
            d.qvel[:first_flex_dof] = 0
            mujoco.mj_step(m,d)
            if (step+1)%round(.02/m.opt.timestep):
                continue
            edge = d.flexvert_xpos[64:66].mean(axis=0)
            row28 = d.flexvert_xpos[56:58].mean(axis=0)
            forearm_contacts = 0
            for c in d.contact:
                if not any(f>=0 for f in c.flex):
                    continue
                bodies = [m.body(m.geom_bodyid[g]).name for g in c.geom if g>=0]
                forearm_contacts += 'training_forearm_dynamic' in bodies
                if any(n.startswith(('g1_a/','g1_b/')) for n in bodies):
                    max_robot_pen = max(max_robot_pen,-float(c.dist))
            trace.append({'time_s':float(d.time),
                          'edge_drift_m':float(np.linalg.norm(edge-edge0)),
                          'row28_drift_m':float(np.linalg.norm(row28-row28_0)),
                          'forearm_flex_contacts':forearm_contacts})
        cases.append({'edge_damping':damping,'pin_end':pin_end,
                      'final_edge_drift_m':trace[-1]['edge_drift_m'],
                      'final_row28_drift_m':trace[-1]['row28_drift_m'],
                      'max_robot_flex_penetration_m':max_robot_pen,
                      'trace':trace})
report = {'scope':'Passive cloth relaxation with all original scene bodies externally fixed; diagnostic only, not robot execution',
          'scene':scene.name,'duration_s':.5,'cases':cases}
(OUT/'joint_flex_relaxation_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in c.items() if k!='trace'} for c in cases]))
