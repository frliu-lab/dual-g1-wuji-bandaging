"""Static contact geometry screen at a failed long-hold frame.

This does not establish dynamic support quality; candidates require a full replay.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
report=json.loads((OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_4b90be62fcdf.json').read_text())
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
data=mujoco.MjData(model)
frame=min(report['trace'],key=lambda r:abs(r['time_s']-17.6))
base=np.asarray(frame['qpos'])
flex=model.joint('g1_a/wuji_left/l_pinky_mcp_flex').qposadr[0]
abd=model.joint('g1_a/wuji_left/l_pinky_mcp_abd').qposadr[0]
rows=[]
for f in np.linspace(.2,1.54,15):
    for a in np.linspace(-.65,.5,13):
        data.qpos[:]=base
        data.qpos[flex]=f
        data.qpos[abd]=a
        mujoco.mj_forward(model,data)
        contacts=[]
        for c in data.contact:
            names=(model.geom(c.geom1).name,model.geom(c.geom2).name)
            if 'manikin_forearm' not in names:continue
            body=next((model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)
                       if model.body(model.geom_bodyid[g]).name.startswith('g1_a/wuji_left/')),None)
            if body:contacts.append((body,float(c.dist)))
        palm=[v for body,v in contacts if body=='g1_a/wuji_left/l_wrist']
        finger=[v for body,v in contacts if body!='g1_a/wuji_left/l_wrist']
        rows.append({'flex_rad':float(f),'abd_rad':float(a),'palm_contacts':len(palm),
                     'finger_contacts':len(finger),'min_palm_dist_m':min(palm,default=None),
                     'min_finger_dist_m':min(finger,default=None)})
rows.sort(key=lambda r:(r['finger_contacts']>0,-r['palm_contacts'],
                        abs(r['flex_rad']-base[flex])+abs(r['abd_rad']-base[abd])))
summary={'source_time_s':frame['time_s'],'base_flex_rad':float(base[flex]),
         'base_abd_rad':float(base[abd]),'screened':len(rows),
         'palm_only_count':sum(r['palm_contacts']>0 and not r['finger_contacts'] for r in rows),
         'best':rows[:20]}
wrist=[model.joint(name).qposadr[0] for name in
       ('g1_a/left_wrist_roll_joint','g1_a/left_wrist_pitch_joint','g1_a/left_wrist_yaw_joint')]
wrist_rows=[]
for roll in np.linspace(-.16,.16,9):
    for pitch in np.linspace(-.12,.12,7):
        for yaw in np.linspace(-.12,.12,7):
            data.qpos[:]=base
            data.qpos[wrist]+=np.array([roll,pitch,yaw])
            mujoco.mj_forward(model,data)
            contacts=[]
            for c in data.contact:
                if 'manikin_forearm' not in (model.geom(c.geom1).name,model.geom(c.geom2).name):continue
                body=next((model.body(model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)
                           if model.body(model.geom_bodyid[g]).name.startswith('g1_a/wuji_left/')),None)
                if body:contacts.append((body,float(c.dist)))
            palm=[v for body,v in contacts if body=='g1_a/wuji_left/l_wrist']
            finger=[v for body,v in contacts if body!='g1_a/wuji_left/l_wrist']
            wrist_rows.append({'offset_rad':[float(roll),float(pitch),float(yaw)],
                               'palm_contacts':len(palm),'finger_contacts':len(finger),
                               'min_palm_dist_m':min(palm,default=None),
                               'min_finger_dist_m':min(finger,default=None)})
wrist_rows.sort(key=lambda r:(not(r['palm_contacts']>0 and not r['finger_contacts']),
                              sum(abs(x) for x in r['offset_rad'])))
summary['wrist_screened']=len(wrist_rows)
summary['wrist_palm_only_count']=sum(r['palm_contacts']>0 and not r['finger_contacts'] for r in wrist_rows)
summary['best_wrist']=wrist_rows[:20]
(OUT/'joint_support_pinky_clearance_screen.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary))
