"""Screen A-left ring-finger retraction against the dynamic terminal forearm pose."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p73_root0p55_spoolz0p702_variant.xml'))
data=mujoco.MjData(model)
q=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_final_state.npz')['qpos'].copy()
names=['g1_a/wuji_left/l_ring_finger_mcp_flex','g1_a/wuji_left/l_ring_finger_mcp_abd',
       'g1_a/wuji_left/l_ring_finger_pip','g1_a/wuji_left/l_ring_finger_dip']
adr=np.array([model.joint(name).qposadr[0] for name in names])
rows=[]
for flex in np.linspace(-.6,1.2,10):
    for abd in np.linspace(-.6,.6,9):
        for pip in (-.2,.2,.6):
            candidate=q.copy();candidate[adr[:3]]=(flex,abd,pip)
            data.qpos[:]=candidate;mujoco.mj_forward(model,data)
            ring_contacts=[];deep_other=[];palm_contact=False
            for c in data.contact:
                if c.geom1<0 or c.geom2<0:continue
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                if 'manikin_forearm' in geoms and 'g1_a/wuji_left/l_wrist' in bodies:
                    palm_contact=True
                if 'manikin_forearm' in geoms and any('g1_a/wuji_left/l_ring_' in b for b in bodies):
                    ring_contacts.append(float(c.dist))
                if 'floor' not in geoms and c.dist<-.003 and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                    if not any('g1_a/wuji_left/l_ring_' in b for b in bodies):
                        deep_other.append((bodies,float(c.dist)))
            rows.append({'mcp_flex':float(flex),'mcp_abd':float(abd),'pip':float(pip),
                         'ring_contact_count':len(ring_contacts),
                         'ring_min_distance_m':min(ring_contacts) if ring_contacts else None,
                         'palm_contact':palm_contact,'deep_other_count':len(deep_other)})
eligible=[r for r in rows if r['ring_contact_count']==0 and r['palm_contact'] and not r['deep_other_count']]
eligible.sort(key=lambda r:abs(r['mcp_flex']-q[adr[0]])+abs(r['mcp_abd']-q[adr[1]])+abs(r['pip']-q[adr[2]]))
wrist_names=['g1_a/left_wrist_roll_joint','g1_a/left_wrist_pitch_joint','g1_a/left_wrist_yaw_joint']
wrist_adr=np.array([model.joint(name).qposadr[0] for name in wrist_names])
wrist_rows=[]
for roll in np.linspace(-.18,.18,9):
    for pitch in np.linspace(-.18,.18,9):
        for yaw in np.linspace(-.18,.18,9):
            candidate=q.copy();candidate[wrist_adr]+=np.array([roll,pitch,yaw])
            data.qpos[:]=candidate;mujoco.mj_forward(model,data)
            palm=False;ring=0;deep=0
            for c in data.contact:
                if c.geom1<0 or c.geom2<0:continue
                bodies=(model.body(model.geom_bodyid[c.geom1]).name,
                        model.body(model.geom_bodyid[c.geom2]).name)
                geoms=(model.geom(c.geom1).name,model.geom(c.geom2).name)
                if 'manikin_forearm' in geoms and 'g1_a/wuji_left/l_wrist' in bodies:palm=True
                if 'manikin_forearm' in geoms and any('g1_a/wuji_left/l_ring_' in b for b in bodies):ring+=1
                if 'floor' not in geoms and c.dist<-.003 and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):deep+=1
            wrist_rows.append({'roll_delta':float(roll),'pitch_delta':float(pitch),'yaw_delta':float(yaw),
                               'palm_contact':palm,'ring_contacts':ring,'deep_contacts':deep})
wrist_eligible=[r for r in wrist_rows if r['palm_contact'] and r['ring_contacts']==0 and r['deep_contacts']==0]
wrist_eligible.sort(key=lambda r:r['roll_delta']**2+r['pitch_delta']**2+r['yaw_delta']**2)
report={'scope':'Static ring-finger clearance at one dynamic terminal state, not dynamic force validation',
        'baseline_ring_qpos_rad':q[adr].tolist(),'tested':len(rows),
        'contact_free_palm_supported_count':len(eligible),
        'ring_contact_free_count':sum(r['ring_contact_count']==0 for r in rows),
        'palm_contact_count':sum(r['palm_contact'] for r in rows),
        'no_deep_other_count':sum(r['deep_other_count']==0 for r in rows),
        'nearest_candidates':eligible[:15],
        'lowest_contact_candidates':sorted(rows,key=lambda r:(r['ring_contact_count'],
                                                                abs(r['mcp_flex']-q[adr[0]])+abs(r['mcp_abd']-q[adr[1]])))[:10],
        'wrist_tested':len(wrist_rows),'wrist_eligible_count':len(wrist_eligible),
        'nearest_wrist_candidates':wrist_eligible[:15]}
(OUT/'joint_support_ring_clearance_screen.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
