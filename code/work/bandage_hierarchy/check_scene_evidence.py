"""A contact fixture verifies that force-bearing hand contact is recognized."""
import mujoco
import xml.etree.ElementTree as ET
from scene_evidence import hand_contacts
from guided_flex_pilot import make_guided_xml

XML = '''<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="manikin_forearm" type="sphere" size=".05" pos="0 0 0"/>
    <body name="g1_a/wuji_left/finger" pos="0 0 .09">
      <freejoint/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
</mujoco>'''
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
assert data.ncon > 0
assert hand_contacts(model, data, .01) == (True, False, False, False)
print('contact evidence fixture passed: force-bearing A-left contact only')

root = ET.fromstring(make_guided_xml())
world = root.find('worldbody')
finger = ET.SubElement(world, 'body', {'name': 'g1_b/wuji_right/finger', 'pos': '0 .050 .12'})
ET.SubElement(finger, 'geom', {'name': 'guide_finger_contact', 'type': 'sphere',
                              'size': '.008', 'rgba': '.1 .3 .8 1'})
model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
contacts = hand_contacts(model, data, .01)
assert contacts[3] and not contacts[0] and not contacts[1] and not contacts[2], contacts
print('flex contact evidence fixture passed: force-bearing B-right flex contact only')
