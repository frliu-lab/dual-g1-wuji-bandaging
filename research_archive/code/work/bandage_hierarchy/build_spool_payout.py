"""Build a dual-G1 scene with cloth initially wound on a freely rotating core."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

OUT = Path(__file__).resolve().parents[2] / 'outputs/wuji_multiagent/bandage_v0'
SOURCE = OUT / 'joint_dual_g1_pinchtab_extended20mm_reinforced_hem.xml'
TARGET = OUT / 'joint_dual_g1_wound_spool.xml'


def build():
    root = ET.parse(SOURCE).getroot()
    world = root.find('worldbody')
    spool = world.find("body[@name='bandage_spool']")
    drum = ET.SubElement(spool, 'body', name='payout_drum')
    ET.SubElement(drum, 'joint', name='payout_axle', type='hinge', axis='0 0 1',
                  limited='false', damping='.00005', armature='.00001')
    for name in ('bandage_roll', 'bandage_core'):
        geom = spool.find(f"geom[@name='{name}']")
        spool.remove(geom)
        drum.append(geom)
    # The flexible strip supplies the outer layers. Keep the rigid core underneath.
    drum.find("geom[@name='bandage_roll']").set('size', '.019 .015')
    world.remove(world.find("body[@name='bandage_reinforced_hem']"))
    root.remove(root.find('tendon'))
    root.remove(root.find('equality'))
    contact = root.find('contact')
    for exclusion in list(contact):
        if exclusion.get('name','').startswith('bandage_hem_internal'):
            contact.remove(exclusion)
    strip = world.find('flexcomp')
    world.remove(strip)
    drum.append(strip)
    theta = np.linspace(0., 4*np.pi, 97)
    radius = .0205 + .003 * theta/(2*np.pi)
    xy = radius[:, None] * np.stack((np.cos(theta), np.sin(theta)), axis=1)
    xy = np.vstack((xy, xy[-1]+np.arange(1,13)[:,None]*np.array([0,.003])))
    points = [(x,y,z) for x,y in xy for z in (-.009,.009)]
    elements = [(2*i,2*i+1,2*i+2,2*i+1,2*i+3,2*i+2) for i in range(len(xy)-1)]
    strip.set('point', ' '.join(f'{v:.8f}' for point in points for v in point))
    strip.set('element', ' '.join(str(v) for row in elements for v in row))
    strip.find('contact').set('selfcollide','auto')
    strip.find('edge').attrib.update(solref='.003 1', solimp='.99 .999 .0001')
    # Two inner vertices are fixed to the drum by the flexcomp parent.
    assert strip.find('pin').get('id') == '0 1'
    return root


if __name__ == '__main__':
    scene = build()
    ET.ElementTree(scene).write(TARGET, encoding='utf-8')
    model = mujoco.MjModel.from_xml_path(str(TARGET))
    print(TARGET.name, model.nflexvert, model.joint('payout_axle').id)
