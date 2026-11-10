"""Build an isolated dynamic training-manikin forearm variant for support research."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
root=ET.parse(OUT/'scene.xml').getroot()
dummy=next(b for b in root.iter('body') if b.get('name')=='training_manikin')
forearm_geom=next(g for g in dummy.findall('geom') if g.get('name')=='manikin_forearm')
dummy.remove(forearm_geom)
wound=next(g for g in dummy.findall('geom') if g.get('name')=='wound_target_visual')
dummy.remove(wound)
elbow=np.array([-.12,-.32,.90])
arm=ET.SubElement(dummy,'body',{'name':'training_forearm_dynamic','pos':'-.12 -.32 .90'})
ET.SubElement(arm,'joint',{'name':'training_elbow_hinge','type':'hinge','axis':'0 1 0',
                           'range':'-0.35 0.35','damping':'0.6','armature':'0.01',
                           'stiffness':'1.0','springref':'0'})
forearm_geom.set('fromto','0 0 0 .34 -.17 -.04')
forearm_geom.set('mass','.8')
arm.append(forearm_geom)
wound_pos=np.fromstring(wound.get('pos'),sep=' ')-elbow
wound.set('pos',' '.join(f'{x:.9g}' for x in wound_pos))
arm.append(wound)
for site in list(dummy.findall('site')):
    if site.get('name','').startswith('bandage_route_'):
        pos=np.fromstring(site.get('pos'),sep=' ')-elbow
        site.set('pos',' '.join(f'{x:.9g}' for x in pos))
        dummy.remove(site);arm.append(site)
variant=OUT/'dynamic_forearm_variant.xml'
ET.ElementTree(root).write(variant,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(variant))
d=mujoco.MjData(m)
mujoco.mj_forward(m,d)
jid=m.joint('training_elbow_hinge').id
qadr=int(m.jnt_qposadr[jid]);vadr=int(m.jnt_dofadr[jid])
initial_forearm=m.geom('manikin_forearm').id
initial_position=d.geom_xpos[initial_forearm].copy()
initial_route=float(d.ten_length[m.tendon('bandage_route_centerline').id])
samples=[]
for _ in range(round(1./m.opt.timestep)):
    mujoco.mj_step(m,d)
    if len(samples)==0 or d.time-samples[-1]['time_s']>=.05:
        samples.append({'time_s':float(d.time),'elbow_rad':float(d.qpos[qadr]),
                        'elbow_speed_rad_s':float(d.qvel[vadr]),
                        'forearm_center_m':d.geom_xpos[initial_forearm].tolist()})
final_route=float(d.ten_length[m.tendon('bandage_route_centerline').id])
report={'scope':'Isolated dynamic training-forearm mechanism; robots uncontrolled during this material check',
        'model_loaded':True,'forearm_mass_kg':.8,
        'initial_elbow_rad':0.,'final_elbow_rad':float(d.qpos[qadr]),
        'max_abs_elbow_rad':max(abs(x['elbow_rad']) for x in samples),
        'initial_forearm_center_m':initial_position.tolist(),
        'final_forearm_center_m':d.geom_xpos[initial_forearm].tolist(),
        'initial_route_length_m':initial_route,'final_route_length_m':final_route,
        'finite_state':bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))),
        'samples':samples}
(OUT/'dynamic_forearm_variant_check.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
