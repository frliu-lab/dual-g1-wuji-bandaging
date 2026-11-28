"""Measure material winding on a stationary cylinder under prescribed handle motion."""
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from build_spool_payout import build, OUT

spool = build().find(".//body[@name='bandage_spool']")
spool.remove(spool.find('freejoint'))
spool.set('pos','0 0 0')
root = ET.Element('mujoco')
ET.SubElement(root,'option',timestep='.001',gravity='0 0 0',integrator='implicitfast')
world = ET.SubElement(root,'worldbody')
carrier = ET.SubElement(world,'body',name='carrier',mocap='true')
carrier.append(spool)
points = np.fromstring(spool.find('.//flexcomp').get('point'),sep=' ').reshape(-1,3)
tail = points[-2:].mean(axis=0)
center = tail[:2]+[.049,0.]
ET.SubElement(world,'geom',name='mandrel',type='cylinder',
              pos=f'{center[0]} {center[1]} 0',size='.048 .04')
eq = ET.SubElement(root,'equality')
for vertex in (len(points)-2,len(points)-1):
    ET.SubElement(eq,'connect',body1=f'joint_bandage_strip_{vertex}',anchor='0 0 0',
                  solref='.003 1',solimp='.99 .999 .0001')
m = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
d = mujoco.MjData(m)
samples = []
for step in range(3000):
    u = min(step/2500.,1.)
    angle = 2*np.pi*u*u*(3-2*u)
    rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    d.mocap_pos[0,:2] = center - rot@center
    mujoco.mj_step(m,d)
    if (step+1)%250 == 0:
        line = d.flexvert_xpos.reshape(-1,2,3).mean(axis=1)[::-1]
        delta = line[:,:2]-center
        radii = np.linalg.norm(delta,axis=1)
        off = np.flatnonzero((radii > .053)|(radii < .045)|(abs(line[:,2])>.04))
        count = int(off[0]) if len(off) else len(line)
        theta = np.unwrap(np.arctan2(delta[:count,1],delta[:count,0]))
        turns = float((theta[-1]-theta[0])/(2*np.pi)) if count>1 else 0.
        samples.append(dict(time_s=float(d.time), contiguous_turns=turns, contiguous_rows=count,
                            tail_error_mm=float(1000*np.linalg.norm(line[0]-tail))))
report=dict(scope='Prescribed circular handle motion, fixed tail, zero gravity mandrel bench',
            samples=samples,warnings=d.warning.number.tolist())
(OUT/'spool_orbit_bench.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
