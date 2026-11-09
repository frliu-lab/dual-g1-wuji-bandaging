"""Extract role-specific contact evidence from a MuJoCo scene.

The fourth role cannot be verified until the bandage has collision geometry.
"""
from pathlib import Path
import json
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / 'outputs/wuji_multiagent/bandage_v0/scene.xml'
ROLES = (
    ('g1_a/wuji_left/', 'geom', 'manikin_forearm'),
    ('g1_a/wuji_right/', 'geom', 'dressing_pad_contact'),
    ('g1_b/wuji_left/', 'geom', 'bandage_roll'),
    ('g1_b/wuji_right/', 'flex', 'bandage_strip'),
)


def hand_contacts(model: mujoco.MjModel, data: mujoco.MjData,
                  min_normal_force_n: float = .05) -> tuple[bool, bool, bool, bool]:
    if min_normal_force_n <= 0:
        raise ValueError('Contact force threshold must be positive')
    hits = [False]*4
    for i, contact in enumerate(data.contact):
        sides = []
        for side in range(2):
            gid, fid = int(contact.geom[side]), int(contact.flex[side])
            if gid >= 0:
                sides.append(('geom', model.geom(gid).name,
                              model.body(model.geom_bodyid[gid]).name))
            elif fid >= 0:
                sides.append(('flex', mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fid), None))
            else:
                sides.append(('none', None, None))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        if wrench[0] < min_normal_force_n:
            continue
        for role, (hand_prefix, target_kind, target_name) in enumerate(ROLES):
            for hand_side in (0, 1):
                kind, name, body = sides[hand_side]
                other_kind, other_name, _ = sides[1-hand_side]
                if (kind == 'geom' and body and body.startswith(hand_prefix)
                        and other_kind == target_kind and other_name == target_name):
                    hits[role] = True
    return tuple(hits)


if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    contacts = hand_contacts(model, data)
    result = {
        'scope': 'Actual MuJoCo contact evidence at initial frame; no motion or completed grasp',
        'role_order': ['A-left forearm', 'A-right pad', 'B-left spool', 'B-right bandage'],
        'hand_contacts': contacts,
        'bandage_flex_present': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX,
                                                 'bandage_strip') >= 0,
        'wrap_specialist_allowed': all(contacts),
    }
    (SCENE.parent / 'contact_evidence_initial.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    if result['bandage_flex_present'] or any(contacts):
        raise RuntimeError('Unexpected bandage geometry or initial hand contact')
