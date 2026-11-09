"""Create the first loadable two-G1 mannequin bandaging scene.

This is a task environment, not a demonstration or a validated controller.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/wuji_multiagent/four_hand_dynamic_v2/table_doorway/scene.xml'
DEST = ROOT / 'outputs/wuji_multiagent/bandage_v0'


def geom(parent, name, kind, **attrs):
    return ET.SubElement(parent, 'geom', {'name': name, 'type': kind, **{k: str(v) for k, v in attrs.items()}})


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
    geom(world, 'manikin_chair_seat', 'box', pos='0 .48 .53', size='.24 .23 .04', rgba='.25 .35 .45 1')
    geom(world, 'manikin_chair_back', 'box', pos='0 .68 .78', size='.24 .035 .27', rgba='.25 .35 .45 1')
    geom(world, 'care_station_backboard', 'box', pos='-.62 1.35 1.28', size='.23 .025 .23', rgba='.92 .94 .92 1', contype='0', conaffinity='0')
    geom(world, 'care_station_cross_v', 'box', pos='-.62 1.315 1.28', size='.035 .004 .13', rgba='.14 .55 .37 1', contype='0', conaffinity='0')
    geom(world, 'care_station_cross_h', 'box', pos='-.62 1.31 1.28', size='.13 .004 .035', rgba='.14 .55 .37 1', contype='0', conaffinity='0')
    geom(world, 'equipment_tray', 'box', pos='.40 -.62 .65', size='.30 .20 .025', rgba='.55 .57 .62 1')
    for sx in ('.16', '.64'):
        geom(world, f'tray_leg_{sx}', 'box', pos=f'{sx} -.62 .32', size='.025 .025 .30', rgba='.35 .37 .42 1')
    geom(world, 'dressing_pad_visual', 'box', pos='.25 -.62 .685', size='.065 .055 .008', rgba='.95 .95 .92 1', contype='0', conaffinity='0')
    spool = ET.SubElement(world, 'body', {'name': 'bandage_spool', 'pos': '.52 -.62 .72'})
    ET.SubElement(spool, 'freejoint', {'name': 'bandage_spool_free'})
    geom(spool, 'bandage_roll', 'cylinder', size='.055 .025', mass='.08', rgba='.96 .94 .88 1', friction='.8 .005 .0001')
    geom(spool, 'bandage_core', 'cylinder', size='.018 .027', mass='.005', rgba='.50 .48 .40 1')
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / 'scene.xml'
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(target))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot_contacts = []
    for c in data.contact:
        a, b = model.geom(c.geom1).name, model.geom(c.geom2).name
        if (a.startswith('g1_') or b.startswith('g1_')) and c.dist < -.001:
            robot_contacts.append((a, b, float(c.dist)))
    report = {
        'scope': 'First task scene only; no bandage path, fabric physics, trained hand specialist, or completed action',
        'model_loaded': True, 'two_g1': all(model.body(f'{r}/pelvis').id >= 0 for r in ('g1_a', 'g1_b')),
        'four_wuji_hands': sum('/wuji_' in model.joint(i).name for i in range(model.njnt)) == 80,
        'manikin_forearm': model.geom('manikin_forearm').id >= 0,
        'initial_deep_robot_contacts': robot_contacts[:20],
        'passed_initial_geometry': not robot_contacts,
    }
    (DEST / 'scene_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    if robot_contacts:
        raise RuntimeError('Initial robot collision in bandage scene')


if __name__ == '__main__':
    main()
