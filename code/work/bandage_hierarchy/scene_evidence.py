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
    ('g1_a/wuji_left/', 'manikin_forearm'),
    ('g1_a/wuji_right/', 'dressing_pad_contact'),
    ('g1_b/wuji_left/', 'bandage_roll'),
    ('g1_b/wuji_right/', 'bandage_strip_contact'),
)


def hand_contacts(model: mujoco.MjModel, data: mujoco.MjData,
                  min_normal_force_n: float = .05) -> tuple[bool, bool, bool, bool]:
    if min_normal_force_n <= 0:
        raise ValueError('Contact force threshold must be positive')
    hits = [False]*4
    for i, contact in enumerate(data.contact):
        ids = (contact.geom1, contact.geom2)
        names = tuple(model.geom(j).name for j in ids)
        bodies = tuple(model.body(model.geom_bodyid[j]).name for j in ids)
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        if wrench[0] < min_normal_force_n:
            continue
        for role, (hand_prefix, target) in enumerate(ROLES):
            if (bodies[0].startswith(hand_prefix) and names[1] == target or
                    bodies[1].startswith(hand_prefix) and names[0] == target):
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
        'bandage_collision_geom_present': any(model.geom(i).name == 'bandage_strip_contact'
                                             for i in range(model.ngeom)),
        'wrap_specialist_allowed': all(contacts),
    }
    (SCENE.parent / 'contact_evidence_initial.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    if result['bandage_collision_geom_present'] or any(contacts):
        raise RuntimeError('Unexpected bandage geometry or initial hand contact')
