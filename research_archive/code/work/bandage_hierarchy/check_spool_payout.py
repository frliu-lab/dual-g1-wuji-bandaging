"""Isolate spool mechanics from grasp control with a supported handle and a test load."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from build_spool_payout import build, OUT

def trial(force):
    spool = build(world_frame=False).find(".//body[@name='bandage_spool']")
    spool.remove(spool.find('freejoint'))
    root = ET.Element('mujoco')
    ET.SubElement(root,'option',timestep='.001',gravity='0 0 0',integrator='implicitfast')
    ET.SubElement(root,'worldbody').append(spool)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    start = data.flexvert_xpos[-2:].mean(axis=0).copy()
    def outside_length():
        line = data.flexvert_xpos.reshape(-1,2,3).mean(axis=1)
        radius = np.linalg.norm((line-data.body('payout_drum').xpos)[:,:2],axis=1)
        crosses = np.flatnonzero(radius <= .028)
        i = int(crosses[-1])
        if i == len(line)-1:
            return 0.
        fraction = (.028-radius[i])/(radius[i+1]-radius[i])
        return float(np.linalg.norm(np.diff(line[i:],axis=0),axis=1).sum()
                     - fraction*np.linalg.norm(line[i+1]-line[i]))
    initial_outside = outside_length()
    max_strain = 0.
    def lengths():
        coords = data.flexvert_xpos
        pairs = model.flex_edge.reshape(-1,2)
        return np.linalg.norm(coords[pairs[:,1]]-coords[pairs[:,0]],axis=1)
    initial = lengths()
    peak = 0.
    for step in range(1000):
        load = force * min(1.,step/100)
        data.xfrc_applied[model.flex_vertbodyid[-2:],1] = load/2
        mujoco.mj_step(model,data)
        peak = max(peak,float(np.max(np.abs(data.qvel))))
        max_strain = max(max_strain,float(np.max(np.abs(lengths()/initial-1))))
    return dict(force_n=force, duration_s=float(data.time),
                end_y_travel_m=float(data.flexvert_xpos[-2:].mean(axis=0)[1]-start[1]),
                axle_rotation_rad=float(data.joint('payout_axle').qpos[0]),
                exposed_length_change_m=outside_length()-initial_outside,
                max_edge_strain=max_strain, peak_generalized_speed=peak,
                warnings=data.warning.number.tolist())

if __name__=='__main__':
    result={'scope':'Zero gravity fixed handle bench, prescribed external pull, no robot execution',
            'trials':[trial(force) for force in (0.,.05,.15)]}
    print(json.dumps(result))
    (OUT/'spool_payout_bench.json').write_text(json.dumps(result,indent=2)+'\n')
