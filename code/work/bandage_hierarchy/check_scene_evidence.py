"""A contact fixture verifies that force-bearing hand contact is recognized."""
import mujoco
from scene_evidence import hand_contacts

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
