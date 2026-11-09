"""Create the first loadable two-G1 mannequin bandaging scene.

This is a task environment, not a demonstration or a validated controller.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/wuji_multiagent/four_hand_dynamic_v2/table_doorway/scene.xml'
DEST = ROOT / 'outputs/wuji_multiagent/bandage_v0'


def geom(parent, name, kind, **attrs):
    return ET.SubElement(parent, 'geom', {'name': name, 'type': kind, **{k: str(v) for k, v in attrs.items()}})


def coords(v):
    return ' '.join(f'{x:.7f}' for x in v)


def main():
    root = ET.parse(SOURCE).getroot()
    root.set('model', 'two_g1_four_wuji_manikin_bandaging_v0')
    world = root.find('worldbody')
    for item in list(world):
        if item.get('name') == 'shared_payload' or item.get('name', '').startswith('door_'):
            world.remove(item)
    # A seated training manikin, visible wound region, and a supported forearm.
    mannequin = ET.SubElement(world, 'body', {'name': 'training_manikin', 'pos': '0 .48 0'})
    geom(mannequin, 'manikin_torso', 'capsule', fromto='0 0 .66 0 0 1.15', size='.16', rgba='.82 .70 .58 1', contype='0', conaffinity='0')
    geom(mannequin, 'manikin_head', 'sphere', pos='0 0 1.33', size='.13', rgba='.82 .70 .58 1', contype='0', conaffinity='0')
    geom(mannequin, 'manikin_neck', 'capsule', fromto='0 0 1.15 0 0 1.24', size='.055', rgba='.82 .70 .58 1', contype='0', conaffinity='0')
    for side, x in (('left', -.085), ('right', .085)):
        geom(mannequin, f'manikin_{side}_thigh', 'capsule', fromto=f'{x} -.06 .60 {x} -.27 .47', size='.065', rgba='.72 .78 .83 1', contype='0', conaffinity='0')
        geom(mannequin, f'manikin_{side}_shin', 'capsule', fromto=f'{x} -.27 .47 {x} -.29 .10', size='.055', rgba='.72 .78 .83 1', contype='0', conaffinity='0')
        geom(mannequin, f'manikin_{side}_foot', 'box', pos=f'{x} -.36 .065', size='.065 .13 .035', rgba='.38 .41 .45 1', contype='0', conaffinity='0')
    geom(mannequin, 'manikin_other_upper_arm', 'capsule', fromto='0 .08 1.09 -.12 .18 .88', size='.055', rgba='.82 .70 .58 1', contype='0', conaffinity='0')
    geom(mannequin, 'manikin_other_forearm', 'capsule', fromto='-.12 .18 .88 -.04 .03 .67', size='.048', rgba='.82 .70 .58 1', contype='0', conaffinity='0')
    geom(mannequin, 'manikin_upper_arm', 'capsule', fromto='0 -.08 1.09 -.12 -.32 .90', size='.055', rgba='.82 .70 .58 1')
    geom(mannequin, 'manikin_forearm', 'capsule', fromto='-.12 -.32 .90 .22 -.49 .86', size='.048', rgba='.82 .70 .58 1', friction='.6 .005 .0001')
    geom(mannequin, 'wound_target_visual', 'cylinder', pos='.055 -.408 .88', size='.052 .049', rgba='.85 .16 .18 .65', contype='0', conaffinity='0')
    # A visible two-turn centerline for testing route length and spool tension.
    # These via sites are fixed to the manikin; this is not deformable cloth.
    p0, p1 = np.array([-.12, -.32, .90]), np.array([.22, -.49, .86])
    axis = (p1-p0) / np.linalg.norm(p1-p0)
    radial_a = np.cross(axis, [0., 0., 1.])
    radial_a /= np.linalg.norm(radial_a)
    radial_b = np.cross(axis, radial_a)
    wrap_points = []
    for i in range(33):
        u = i / 32
        theta = 4*np.pi*u
        point = (p0+p1)/2 + (.065*(u-.5))*axis + .052*(np.cos(theta)*radial_a+np.sin(theta)*radial_b)
        name = f'bandage_route_{i:02d}'
        ET.SubElement(mannequin, 'site', {'name': name, 'pos': coords(point), 'size': '.002', 'rgba': '.9 .85 .7 .5'})
        wrap_points.append((name, point + [0., .48, 0.]))
    geom(world, 'manikin_chair_seat', 'box', pos='0 .48 .53', size='.24 .23 .04', rgba='.25 .35 .45 1')
    geom(world, 'manikin_chair_back', 'box', pos='0 .68 .78', size='.24 .035 .27', rgba='.25 .35 .45 1')
    geom(world, 'care_station_backboard', 'box', pos='-.62 1.35 1.28', size='.23 .025 .23', rgba='.92 .94 .92 1', contype='0', conaffinity='0')
    geom(world, 'care_station_cross_v', 'box', pos='-.62 1.315 1.28', size='.035 .004 .13', rgba='.14 .55 .37 1', contype='0', conaffinity='0')
    geom(world, 'care_station_cross_h', 'box', pos='-.62 1.31 1.28', size='.13 .004 .035', rgba='.14 .55 .37 1', contype='0', conaffinity='0')
    geom(world, 'equipment_tray', 'box', pos='.40 -.62 .65', size='.30 .20 .025', rgba='.55 .57 .62 1')
    for sx in ('.16', '.64'):
        geom(world, f'tray_leg_{sx}', 'box', pos=f'{sx} -.62 .32', size='.025 .025 .30', rgba='.35 .37 .42 1')
    pad = ET.SubElement(world, 'body', {'name': 'dressing_pad', 'pos': '.25 -.62 .695'})
    ET.SubElement(pad, 'freejoint', {'name': 'dressing_pad_free'})
    geom(pad, 'dressing_pad_contact', 'box', size='.065 .055 .008', mass='.012',
         rgba='.95 .95 .92 1', friction='1 .005 .0001')
    ET.SubElement(pad, 'site', {'name': 'bandage_pad_anchor', 'pos': '0 0 .01', 'size': '.003'})
    spool = ET.SubElement(world, 'body', {'name': 'bandage_spool', 'pos': '.52 -.62 .72'})
    ET.SubElement(spool, 'freejoint', {'name': 'bandage_spool_free'})
    geom(spool, 'bandage_roll', 'cylinder', size='.055 .025', mass='.08', rgba='.96 .94 .88 1', friction='.8 .005 .0001')
    geom(spool, 'bandage_core', 'cylinder', size='.018 .027', mass='.005', rgba='.50 .48 .40 1')
    ET.SubElement(spool, 'site', {'name': 'bandage_spool_anchor', 'pos': '0 0 0', 'size': '.003'})
    route_xyz = [np.array([.25, -.62, .705]), *(p for _, p in wrap_points), np.array([.52, -.62, .72])]
    route_length = sum(float(np.linalg.norm(b-a)) for a, b in zip(route_xyz, route_xyz[1:]))
    tendon = root.find('tendon')
    if tendon is None:
        tendon = ET.SubElement(root, 'tendon')
    route = ET.SubElement(tendon, 'spatial', {'name': 'bandage_route_centerline', 'width': '.012',
                                          'rgba': '.94 .88 .73 1', 'stiffness': '2',
                                          'damping': '.02', 'springlength': f'{route_length:.7f}'})
    ET.SubElement(route, 'site', {'site': 'bandage_pad_anchor'})
    for name, _ in wrap_points:
        ET.SubElement(route, 'site', {'site': name})
    ET.SubElement(route, 'site', {'site': 'bandage_spool_anchor'})
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / 'scene.xml'
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(target))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pad_id = model.body('dressing_pad').id
    pad_initial_z = float(data.xpos[pad_id, 2])
    for _ in range(round(.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    pad_settled_z = float(data.xpos[pad_id, 2])
    route_id = model.tendon('bandage_route_centerline').id
    route_length_after = float(data.ten_length[route_id])
    robot_contacts = []
    for c in data.contact:
        a, b = model.geom(c.geom1).name, model.geom(c.geom2).name
        if (a.startswith('g1_') or b.startswith('g1_')) and c.dist < -.001:
            robot_contacts.append((a, b, float(c.dist)))
    report = {
        'scope': 'Two-turn tendon centerline is a geometric proxy; no fabric contact, calibrated tension, trained hand specialist, or completed action',
        'model_loaded': True, 'two_g1': all(model.body(f'{r}/pelvis').id >= 0 for r in ('g1_a', 'g1_b')),
        'four_wuji_hands': sum('/wuji_' in model.joint(i).name for i in range(model.njnt)) == 80,
        'manikin_forearm': model.geom('manikin_forearm').id >= 0,
        'dressing_pad_dynamic': int(model.body('dressing_pad').jntnum[0]) == 1,
        'dressing_pad_initial_z_m': pad_initial_z,
        'dressing_pad_after_half_second_z_m': pad_settled_z,
        'dressing_pad_settled_on_tray': .67 < pad_settled_z < .70,
        'bandage_centerline_two_turns': len(wrap_points) == 33,
        'bandage_route_initial_length_m': route_length,
        'bandage_route_after_half_second_length_m': route_length_after,
        'bandage_route_after_half_second_extension_m': route_length_after-route_length,
        'initial_deep_robot_contacts': robot_contacts[:20],
        'passed_initial_geometry': bool(not robot_contacts and .67 < pad_settled_z < .70 and np.isfinite(route_length_after)),
    }
    (DEST / 'scene_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    if not report['passed_initial_geometry']:
        raise RuntimeError('Initial robot collision or dressing pad failed to settle on tray')


if __name__ == '__main__':
    main()
